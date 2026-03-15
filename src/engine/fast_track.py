"""Fast Track Mode — ultra-short market trading (5-min Up/Down markets).

Runs as a concurrent async task alongside the main trading engine.
Supports two strategies:
  - "llm":  Lightweight LLM analysis on orderbook + price data (no web research)
  - "copy": Accelerated whale copy trading
"""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import time
from dataclasses import dataclass, field
from typing import Any

import structlog

from src.config import BotConfig, FastTrackConfig, is_live_trading_enabled
from src.connectors.polymarket_clob import CLOBClient
from src.connectors.polymarket_gamma import GammaClient, GammaMarket, classify_market_type

log = structlog.get_logger(__name__)


# ── Data Models ──────────────────────────────────────────────────────

@dataclass
class FastTrackCandidate:
    """A market eligible for fast-track trading."""
    market: GammaMarket
    token_id: str = ""
    direction: str = ""        # BUY_YES or BUY_NO
    best_ask: float = 0.0
    minutes_to_expiry: float = 0.0
    strategy_used: str = ""


@dataclass
class FastTrackResult:
    """Result of a fast-track cycle."""
    cycle_id: int = 0
    discovered: int = 0
    traded: int = 0
    auto_exited: int = 0
    errors: list[str] = field(default_factory=list)


# ── Market Discovery ────────────────────────────────────────────────

async def discover_fast_track_markets(
    cfg: FastTrackConfig,
) -> list[GammaMarket]:
    """Fetch and filter markets eligible for fast-track trading."""
    client = GammaClient()
    try:
        # Fetch newest crypto markets
        markets = await client.list_markets(
            limit=100,
            order="startDate",
            ascending=False,
            category="crypto",
        )
    finally:
        await client.close()

    now = dt.datetime.now(dt.timezone.utc)
    keywords = [kw.lower() for kw in cfg.market_keywords]
    eligible: list[GammaMarket] = []

    for m in markets:
        # Keyword filter
        q = m.question.lower()
        if not any(kw in q for kw in keywords):
            continue

        # Expiry filter
        if not m.end_date:
            continue
        end = m.end_date
        if end.tzinfo is None:
            end = end.replace(tzinfo=dt.timezone.utc)
        minutes_left = (end - now).total_seconds() / 60

        if minutes_left < cfg.min_market_duration_minutes:
            continue  # too close to expiry
        if minutes_left > cfg.max_market_duration_minutes:
            continue  # too far out

        # Must have tokens
        if not m.tokens:
            continue

        eligible.append(m)

    log.info("fast_track.discovery",
             total=len(markets), eligible=len(eligible),
             keywords=keywords[:3])
    return eligible


# ── LLM Strategy ────────────────────────────────────────────────────

async def llm_analyze(
    market: GammaMarket,
    token_id: str,
    cfg: FastTrackConfig,
    config: BotConfig,
) -> dict[str, Any] | None:
    """Run lightweight LLM analysis on orderbook + price action.

    Returns {"probability": float, "confidence": str, "reasoning": str}
    or None if analysis fails/insufficient.
    """
    clob = CLOBClient()
    try:
        ob = await clob.get_orderbook(token_id)
        trades = await clob.get_trade_history(token_id, limit=50)
    except Exception as e:
        log.warning("fast_track.llm_data_error", error=str(e))
        return None
    finally:
        await clob.close()

    if not trades:
        return None

    # Compute technical signals
    prices = [t.price for t in trades if t.price > 0]
    if not prices:
        return None

    mid_price = (ob.best_bid + ob.best_ask) / 2 if ob.best_bid > 0 and ob.best_ask > 0 else prices[0]
    vwap = sum(t.price * t.size for t in trades) / max(sum(t.size for t in trades), 0.001)
    momentum = (prices[0] - prices[-1]) / max(prices[-1], 0.001) if len(prices) > 1 else 0

    # Bid/ask imbalance
    bid_depth = sum(level[1] for level in ob.bids[:5]) if ob.bids else 0
    ask_depth = sum(level[1] for level in ob.asks[:5]) if ob.asks else 0
    total_depth = bid_depth + ask_depth
    imbalance = (bid_depth - ask_depth) / total_depth if total_depth > 0 else 0

    # Recent trade direction
    buys = sum(1 for t in trades[:20] if t.side.lower() == "buy")
    sells = sum(1 for t in trades[:20] if t.side.lower() == "sell")

    prompt = f"""You are analyzing a 5-minute crypto prediction market on Polymarket.

QUESTION: {market.question}

MARKET DATA:
- Mid price: {mid_price:.4f}
- Best bid: {ob.best_bid:.4f} | Best ask: {ob.best_ask:.4f}
- Bid/Ask imbalance: {imbalance:+.3f} (positive = more buy pressure)
- VWAP (last 50 trades): {vwap:.4f}
- Price momentum: {momentum:+.4f}
- Recent trades: {buys} buys vs {sells} sells (last 20)
- Total bid depth (top 5): {bid_depth:.1f}
- Total ask depth (top 5): {ask_depth:.1f}

Analyze the market microstructure and predict the probability of YES outcome.
Consider: order flow imbalance, momentum, price vs VWAP.

Respond with ONLY a JSON object:
{{"probability": 0.XX, "confidence": "LOW|MEDIUM|HIGH", "reasoning": "one sentence"}}"""

    try:
        import anthropic
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=config.forecasting.primary_model,
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}],
            timeout=cfg.llm_timeout_secs,
        )
        text = response.content[0].text.strip()
        # Parse JSON from response
        if "{" in text:
            json_str = text[text.index("{"):text.rindex("}") + 1]
            result = json.loads(json_str)
            return result
    except Exception as e:
        log.warning("fast_track.llm_error", error=str(e))

    return None


# ── Auto-Exit Monitor ────────────────────────────────────────────────

async def check_auto_exits(
    db: Any,
    cfg: FastTrackConfig,
    execution_cfg: Any,
) -> int:
    """Check fast-track positions for expiry-based auto-exits.

    Returns number of positions auto-exited.
    """
    if not db:
        return 0

    positions = db.get_open_positions()
    ft_positions = [p for p in positions if getattr(p, "fast_track", 0)]
    if not ft_positions:
        return 0

    now = dt.datetime.now(dt.timezone.utc)
    exited = 0

    for pos in ft_positions:
        # Check if market is about to expire
        # end_date stored in position metadata or we look it up
        try:
            # Try to get end_date from fast_track_log
            row = db.conn.execute(
                "SELECT end_date FROM fast_track_log WHERE market_id = ? ORDER BY created_at DESC LIMIT 1",
                (pos.market_id,)
            ).fetchone()
            if not row or not row["end_date"]:
                continue

            end_date = dt.datetime.fromisoformat(row["end_date"].replace("Z", "+00:00"))
            secs_left = (end_date - now).total_seconds()

            if secs_left > cfg.auto_exit_before_expiry_secs:
                continue

            log.info("fast_track.auto_exit",
                     market=pos.market_id[:8],
                     secs_left=round(secs_left))

            # Fetch best bid for sell price
            clob = CLOBClient()
            sell_price = pos.current_price if pos.current_price > 0 else 0.5
            try:
                ob = await clob.get_orderbook(pos.token_id)
                if ob.best_bid > 0:
                    sell_price = ob.best_bid
            except Exception:
                pass
            finally:
                await clob.close()

            # Build sell order
            from src.execution.order_builder import OrderSpec
            from src.execution.order_router import OrderRouter

            is_dry = execution_cfg.dry_run or cfg.simulate_only
            limit_price = round(sell_price * (1 - 0.02), 4)  # tight slippage for auto-exit
            sell_order = OrderSpec(
                order_id=f"ft-exit-{pos.market_id[:8]}-{int(time.time())}",
                market_id=pos.market_id,
                token_id=pos.token_id,
                side="SELL",
                order_type="GTC",
                price=limit_price,
                size=pos.size,
                stake_usd=pos.stake_usd,
                ttl_secs=30,
                dry_run=is_dry,
                metadata={"market_price": sell_price, "fast_track": True},
            )

            clob2 = CLOBClient()
            router = OrderRouter(clob2, execution_cfg)
            try:
                result = await router.submit_order(sell_order)
                if result.status != "failed":
                    pnl = round((sell_price - pos.entry_price) * pos.size, 4)
                    db.archive_position(
                        pos=pos, exit_price=sell_price,
                        pnl=pnl, close_reason="FT_AUTO_EXIT",
                    )
                    db.remove_position(pos.market_id)

                    # Update fast_track_log
                    db.conn.execute("""
                        UPDATE fast_track_log SET
                            exit_price = ?, pnl = ?, auto_exited = 1,
                            status = 'closed', closed_at = ?
                        WHERE market_id = ? AND status != 'closed'
                    """, (sell_price, pnl,
                          dt.datetime.now(dt.timezone.utc).isoformat(),
                          pos.market_id))
                    db.conn.commit()

                    db.insert_alert(
                        "info",
                        f"⚡ FAST EXIT: Auto-exited {pos.question[:40]} "
                        f"({secs_left:.0f}s before expiry, P&L ${pnl:+.2f})",
                        "fast_track",
                    )
                    exited += 1
            finally:
                await clob2.close()

        except Exception as e:
            log.warning("fast_track.auto_exit_error",
                        market=pos.market_id[:8], error=str(e))

    return exited


# ── Fast Track Cycle Runner ─────────────────────────────────────────

async def run_fast_track_cycle(
    cycle_id: int,
    config: BotConfig,
    db: Any,
    latest_scan_result: Any | None = None,
) -> FastTrackResult:
    """Execute one fast-track cycle."""
    cfg = config.fast_track
    result = FastTrackResult(cycle_id=cycle_id)

    try:
        # 1. Discover eligible markets
        markets = await discover_fast_track_markets(cfg)
        result.discovered = len(markets)

        if not markets:
            # Still check auto-exits even if no new markets
            result.auto_exited = await check_auto_exits(db, cfg, config.execution)
            return result

        # 2. Check position limits
        if db:
            positions = db.get_open_positions()
            ft_count = sum(1 for p in positions if getattr(p, "fast_track", 0))
            if ft_count >= cfg.max_open_positions:
                log.info("fast_track.max_positions", count=ft_count, max=cfg.max_open_positions)
                result.auto_exited = await check_auto_exits(db, cfg, config.execution)
                return result

            # Skip markets we already have positions in
            held_markets = {p.market_id for p in positions}
        else:
            held_markets = set()

        # 3. Process each eligible market
        for market in markets:
            if market.id in held_markets:
                continue

            try:
                await _process_fast_track_market(
                    market, cfg, config, db, latest_scan_result, result
                )
            except Exception as e:
                result.errors.append(f"{market.question[:30]}: {e}")
                log.warning("fast_track.market_error",
                            market=market.question[:30], error=str(e))

            # Recheck position limit
            if db:
                ft_count = sum(1 for p in db.get_open_positions()
                               if getattr(p, "fast_track", 0))
                if ft_count >= cfg.max_open_positions:
                    break

        # 4. Auto-exit expiring positions
        result.auto_exited = await check_auto_exits(db, cfg, config.execution)

    except Exception as e:
        result.errors.append(str(e))
        log.error("fast_track.cycle_error", error=str(e))

    log.info("fast_track.cycle_complete",
             cycle=cycle_id, discovered=result.discovered,
             traded=result.traded, auto_exited=result.auto_exited)
    return result


async def _process_fast_track_market(
    market: GammaMarket,
    cfg: FastTrackConfig,
    config: BotConfig,
    db: Any,
    latest_scan_result: Any | None,
    result: FastTrackResult,
) -> None:
    """Process a single fast-track market based on strategy."""

    # Get token IDs
    yes_token = ""
    no_token = ""
    for tok in market.tokens:
        if tok.outcome.lower() == "yes":
            yes_token = tok.token_id
        elif tok.outcome.lower() == "no":
            no_token = tok.token_id
    if not yes_token:
        yes_token = market.tokens[0].token_id if market.tokens else ""
    if not yes_token:
        return

    # Compute minutes to expiry
    now = dt.datetime.now(dt.timezone.utc)
    end = market.end_date
    if end and end.tzinfo is None:
        end = end.replace(tzinfo=dt.timezone.utc)
    mins_left = (end - now).total_seconds() / 60 if end else 0

    if cfg.strategy == "llm":
        await _fast_track_llm(
            market, yes_token, no_token, mins_left, cfg, config, db, result
        )
    elif cfg.strategy == "copy":
        await _fast_track_copy(
            market, yes_token, no_token, mins_left, cfg, config, db,
            latest_scan_result, result
        )


async def _fast_track_llm(
    market: GammaMarket,
    yes_token: str,
    no_token: str,
    mins_left: float,
    cfg: FastTrackConfig,
    config: BotConfig,
    db: Any,
    result: FastTrackResult,
) -> None:
    """LLM strategy: analyze orderbook + price action, trade if edge."""
    analysis = await llm_analyze(market, yes_token, cfg, config)
    if not analysis:
        return

    prob = float(analysis.get("probability", 0.5))
    confidence = analysis.get("confidence", "LOW")
    if confidence == "LOW":
        return

    # Get live price
    clob = CLOBClient()
    try:
        ob = await clob.get_orderbook(yes_token)
    finally:
        await clob.close()

    market_price = ob.best_ask if ob.best_ask > 0 else 0.5
    implied = market_price

    # Compute edge
    edge = prob - implied
    if edge < cfg.min_edge:
        # Check NO side
        no_edge = (1 - prob) - (1 - implied)
        if no_edge < cfg.min_edge or not no_token:
            log.info("fast_track.llm_no_edge",
                     market=market.question[:30],
                     prob=round(prob, 3), implied=round(implied, 3))
            return
        # Trade NO
        token_id = no_token
        direction = "BUY_NO"
        price = 1 - implied
        clob2 = CLOBClient()
        try:
            ob2 = await clob2.get_orderbook(no_token)
            if ob2.best_ask > 0:
                price = ob2.best_ask
        finally:
            await clob2.close()
    else:
        token_id = yes_token
        direction = "BUY_YES"
        price = market_price

    await _submit_fast_track_order(
        market=market,
        token_id=token_id,
        direction=direction,
        price=price,
        mins_left=mins_left,
        strategy="llm",
        cfg=cfg,
        config=config,
        db=db,
        result=result,
        extra_meta={"llm_prob": prob, "llm_confidence": confidence,
                     "reasoning": analysis.get("reasoning", "")},
    )


async def _fast_track_copy(
    market: GammaMarket,
    yes_token: str,
    no_token: str,
    mins_left: float,
    cfg: FastTrackConfig,
    config: BotConfig,
    db: Any,
    latest_scan_result: Any | None,
    result: FastTrackResult,
) -> None:
    """Copy strategy: check if whales are trading this market."""
    if not latest_scan_result:
        return

    signals = latest_scan_result.conviction_signals or []
    deltas = latest_scan_result.deltas or []

    # Check if any whale has a position in this market
    matched_signal = None
    for sig in signals:
        if (sig.market_slug in market.id or
            sig.market_slug in (market.slug or "") or
            sig.market_slug in market.question.lower()):
            matched_signal = sig
            break

    # Also check for recent whale deltas
    matched_delta = None
    for delta in deltas:
        if (delta.market_slug in market.id or
            delta.market_slug in (market.slug or "")):
            if delta.action == "NEW_ENTRY":
                matched_delta = delta
                break

    if not matched_signal and not matched_delta:
        return

    # Determine direction from whale signal
    if matched_delta:
        outcome = matched_delta.outcome.lower()
        whale_name = matched_delta.wallet_name
    elif matched_signal:
        outcome = matched_signal.outcome.lower()
        whale_name = ", ".join(matched_signal.whale_names[:2])
    else:
        return

    if outcome == "yes":
        token_id = yes_token
        direction = "BUY_YES"
    else:
        token_id = no_token or yes_token
        direction = "BUY_NO"

    # Get live price
    clob = CLOBClient()
    try:
        ob = await clob.get_orderbook(token_id)
        price = ob.best_ask if ob.best_ask > 0 and ob.best_ask < 1.0 else 0.5
    finally:
        await clob.close()

    await _submit_fast_track_order(
        market=market,
        token_id=token_id,
        direction=direction,
        price=price,
        mins_left=mins_left,
        strategy="copy",
        cfg=cfg,
        config=config,
        db=db,
        result=result,
        extra_meta={"whale_source": whale_name},
    )


# ── Order Submission ─────────────────────────────────────────────────

async def _submit_fast_track_order(
    *,
    market: GammaMarket,
    token_id: str,
    direction: str,
    price: float,
    mins_left: float,
    strategy: str,
    cfg: FastTrackConfig,
    config: BotConfig,
    db: Any,
    result: FastTrackResult,
    extra_meta: dict | None = None,
) -> None:
    """Build and submit a fast-track order."""
    from src.execution.order_builder import OrderSpec
    from src.execution.order_router import OrderRouter

    is_dry = config.execution.dry_run or cfg.simulate_only
    slippage = config.execution.slippage_tolerance
    limit_price = round(price * (1 + slippage), 4)
    size = round(cfg.max_stake_per_trade / price, 2) if price > 0 else 0
    if size <= 0:
        return

    meta = {"market_price": price, "fast_track": True, "strategy": strategy}
    if extra_meta:
        meta.update(extra_meta)

    order = OrderSpec(
        order_id=f"ft-{strategy[:3]}-{market.id[:8]}-{int(time.time())}",
        market_id=market.id,
        token_id=token_id,
        side="BUY",
        order_type="GTC",
        price=limit_price,
        size=size,
        stake_usd=cfg.max_stake_per_trade,
        ttl_secs=int(mins_left * 60) - cfg.auto_exit_before_expiry_secs,
        dry_run=is_dry,
        metadata=meta,
    )

    clob = CLOBClient()
    router = OrderRouter(clob, config.execution)
    try:
        fill = await router.submit_order(order)
        if fill.status == "failed":
            return

        log.info("fast_track.trade_executed",
                 strategy=strategy, market=market.question[:50],
                 direction=direction, price=fill.fill_price,
                 status=fill.status, simulated=is_dry)

        if db:
            from src.storage.models import TradeRecord, PositionRecord

            mtype = market.market_type or classify_market_type(market.question)
            end_iso = market.end_date.isoformat() if market.end_date else ""

            db.insert_trade(TradeRecord(
                id=order.order_id, order_id=order.order_id,
                market_id=market.id, token_id=token_id,
                side=direction, price=fill.fill_price,
                size=fill.fill_size, stake_usd=cfg.max_stake_per_trade,
                status=fill.status, dry_run=is_dry,
            ))
            db.upsert_position(PositionRecord(
                market_id=market.id, token_id=token_id,
                direction=direction, entry_price=fill.fill_price,
                size=fill.fill_size, stake_usd=cfg.max_stake_per_trade,
                current_price=fill.fill_price, pnl=0.0,
                question=market.question[:200], market_type=mtype,
                fast_track=1,
            ))

            # Log to fast_track_log
            db.conn.execute("""
                INSERT INTO fast_track_log
                    (market_id, question, strategy, direction, entry_price,
                     stake_usd, is_simulated, end_date, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                market.id, market.question[:200], strategy, direction,
                fill.fill_price, cfg.max_stake_per_trade,
                1 if is_dry else 0, end_iso, "open",
                dt.datetime.now(dt.timezone.utc).isoformat(),
            ))
            db.conn.commit()

            db.insert_alert(
                "info",
                f"⚡ FAST TRACK [{strategy.upper()}]: {direction} "
                f"{market.question[:40]} @ {fill.fill_price:.4f} "
                f"(${cfg.max_stake_per_trade:.2f}, {mins_left:.0f}m left)",
                "fast_track",
            )

        result.traded += 1

    finally:
        await clob.close()
