"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

// ── shapes of the JSON the pipeline writes to S3 ──
type Champion = {
  version: number;
  model_type: string;
  test_mae_mw: number;
  test_wape: number;
};
type ForecastRow = {
  timestamp: string;
  predicted_load_mw: number;
};
type Payload = {
  region: string;
  generated_at: string;
  champion: Champion;
  forecast: ForecastRow[];
};

const REGIONS = ["CAL", "TEX", "PJM", "MISO"] as const;
const REGION_NAMES: Record<string, string> = {
  CAL: "California",
  TEX: "Texas (ERCOT)",
  PJM: "Mid-Atlantic",
  MISO: "Midwest",
};

function fmtMW(n: number): string {
  return Math.round(n).toLocaleString() + " MW";
}

function fmtPct(n: number): string {
  return n > 0 ? n.toFixed(2) + "%" : "—";
}

// reused chrome
const PANEL = "bg-panel border border-border rounded-2xl p-6";

function Card({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div className="bg-panel border border-border rounded-2xl px-5 py-[18px]">
      <div className="text-muted text-xs uppercase tracking-wider mb-2">{label}</div>
      <div className={`text-2xl font-bold ${accent ? "text-accent" : ""}`}>{value}</div>
    </div>
  );
}

export default function Home() {
  const [region, setRegion] = useState<string>("CAL");
  const [data, setData] = useState<Payload | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    setData(null);

    // Hits our own server route (the middleman), which reads private S3.
    // Browser never touches AWS directly.
    fetch(`/api/forecast/${region}`, { cache: "no-store" })
      .then((r) => {
        if (!r.ok) throw new Error(`No forecast yet for ${region} (HTTP ${r.status})`);
        return r.json();
      })
      .then((json: Payload) => {
        if (!cancelled) setData(json);
      })
      .catch((e) => {
        if (!cancelled) setError(e.message);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [region]);

  // Prep chart data: everything in UTC (the dataset is UTC end to end). 24h
  // crosses midnight, so anchor the day: weekday on the first tick and at each
  // "00" so the wrap is readable.
  const chartData =
    data?.forecast.map((row, i) => {
      const d = new Date(row.timestamp);
      const hh = d.toLocaleTimeString("en-US", {
        timeZone: "UTC",
        hour: "2-digit",
        hour12: false,
      });
      const day = d.toLocaleDateString("en-US", {
        timeZone: "UTC",
        weekday: "short",
      });
      return {
        hour: i === 0 || hh === "00" ? `${day} ${hh}` : hh,
        mw: row.predicted_load_mw,
      };
    }) ?? [];

  const peak = data ? Math.max(...data.forecast.map((r) => r.predicted_load_mw)) : 0;
  const trough = data ? Math.min(...data.forecast.map((r) => r.predicted_load_mw)) : 0;

  const nextHour = data?.forecast[0] ?? null;
  const nextHourLabel = nextHour
    ? new Date(nextHour.timestamp).toLocaleTimeString("en-US", {
        timeZone: "UTC",
        hour: "2-digit",
        minute: "2-digit",
        hour12: false,
      }) + " UTC"
    : null;

  return (
    <main className="max-w-[1000px] mx-auto px-5 pt-10 pb-20">
      <div className="flex items-baseline gap-3 flex-wrap mb-1.5">
        <h1 className="text-3xl font-bold tracking-tight">
          <span className="text-accent">⚡</span> VoltCast
        </h1>
        <Link href="/how-it-works" className="ml-auto text-accent-2 hover:text-accent text-sm font-semibold">
          How it works →
        </Link>
      </div>
      <p className="text-muted text-[15px] mb-7">
        24-hour-ahead US electricity demand forecasts, powered by a from-scratch Transformer.
      </p>

      {/* region tabs */}
      <div className="flex gap-2 flex-wrap mb-7">
        {REGIONS.map((r) => {
          const active = r === region;
          return (
            <button
              key={r}
              onClick={() => setRegion(r)}
              className={`px-[18px] py-2.5 rounded-[10px] text-sm font-semibold border transition-colors ${
                active
                  ? "bg-accent border-accent text-bg"
                  : "bg-panel border-border text-muted hover:border-accent-2 hover:text-text"
              }`}
            >
              {REGION_NAMES[r]}
            </button>
          );
        })}
      </div>

      {loading && <div className="text-center py-16 text-muted">Loading {REGION_NAMES[region]}…</div>}
      {error && <div className="text-center py-16 text-[#ff6b6b]">{error}</div>}

      {data && !loading && !error && (
        <>
          {/* next-hour hero */}
          {nextHour && (
            <div className="bg-panel border border-accent/40 rounded-2xl px-6 py-5 mb-5 flex items-center justify-between gap-4 flex-wrap">
              <div>
                <div className="text-muted text-xs uppercase tracking-wider mb-1">Next hour load</div>
                <div className="text-4xl font-bold text-accent tabular">
                  {fmtMW(nextHour.predicted_load_mw)}
                </div>
              </div>
              <div className="text-right">
                <div className="text-muted text-xs uppercase tracking-wider mb-1">Forecast for</div>
                <div className="text-lg font-semibold">{nextHourLabel}</div>
              </div>
            </div>
          )}

          {/* champion + stat cards */}
          <div className="grid grid-cols-[repeat(auto-fit,minmax(180px,1fr))] gap-3.5 mb-6">
            <Card
              label="Model"
              value={
                data.champion.model_type.charAt(0).toUpperCase() +
                data.champion.model_type.slice(1)
              }
            />
            <Card label="Version" value={`v${data.champion.version}`} />
            <Card label="Test MAE" value={fmtMW(data.champion.test_mae_mw)} accent />
            <Card label="Test WAPE" value={fmtPct(data.champion.test_wape)} accent />
            <Card label="Peak" value={fmtMW(peak)} />
            <Card label="Trough" value={fmtMW(trough)} />
          </div>

          {/* forecast chart */}
          <div className={PANEL}>
            <h2 className="text-base font-semibold mb-1">
              Next 24 hours — {REGION_NAMES[region]}{" "}
              <span className="text-muted text-[13px] font-normal">(UTC)</span>
            </h2>
            <div className="text-muted text-[13px] mb-5">
              Generated{" "}
              {new Date(data.generated_at).toLocaleString("en-US", {
                timeZone: "UTC",
                dateStyle: "medium",
                timeStyle: "short",
              })}{" "}
              UTC
            </div>
            <ResponsiveContainer width="100%" height={360}>
              <LineChart data={chartData} margin={{ top: 10, right: 20, bottom: 0, left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#232c44" />
                <XAxis dataKey="hour" stroke="#8b95ad" fontSize={12} />
                <YAxis
                  stroke="#8b95ad"
                  fontSize={12}
                  tickFormatter={(v) => `${Math.round(v / 1000)}k`}
                  domain={["auto", "auto"]}
                />
                <Tooltip
                  contentStyle={{
                    background: "#1a2138",
                    border: "1px solid #232c44",
                    borderRadius: 8,
                    color: "#e6ebf5",
                  }}
                  formatter={(v: number) => [fmtMW(v), "Predicted load"]}
                />
                <Line
                  type="monotone"
                  dataKey="mw"
                  stroke="#ffd23f"
                  strokeWidth={2.5}
                  dot={false}
                  activeDot={{ r: 5 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* hourly table — same UTC data as the chart, exact numbers */}
          <div className={`${PANEL} mt-5`}>
            <h2 className="text-base font-semibold mb-1">
              Hourly forecast <span className="text-muted text-[13px] font-normal">(UTC)</span>
            </h2>
            <div className="mt-4 max-h-[420px] overflow-y-auto">
              <table className="w-full border-collapse text-sm">
                <thead>
                  <tr>
                    <th className="text-left text-muted font-medium text-xs uppercase tracking-wide px-3 py-2 sticky top-0 bg-panel border-b border-border">
                      Time
                    </th>
                    <th className="text-right text-muted font-medium text-xs uppercase tracking-wide px-3 py-2 sticky top-0 bg-panel border-b border-border">
                      Predicted load
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {data.forecast.map((row) => {
                    const d = new Date(row.timestamp);
                    const isPeak = row.predicted_load_mw === peak;
                    const isTrough = row.predicted_load_mw === trough;
                    return (
                      <tr key={row.timestamp}>
                        <td className="px-3 py-2 border-b border-border">
                          {d.toLocaleString("en-US", {
                            timeZone: "UTC",
                            weekday: "short",
                            hour: "2-digit",
                            minute: "2-digit",
                            hour12: false,
                          })}
                          {isPeak && (
                            <span className="ml-2 text-[11px] px-1.5 py-px rounded uppercase tracking-wide bg-accent/15 text-accent">
                              peak
                            </span>
                          )}
                          {isTrough && (
                            <span className="ml-2 text-[11px] px-1.5 py-px rounded uppercase tracking-wide bg-muted/15 text-muted">
                              trough
                            </span>
                          )}
                        </td>
                        <td className="px-3 py-2 border-b border-border text-right tabular">
                          {fmtMW(row.predicted_load_mw)}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      <div className="mt-10 text-muted text-[13px] text-center">
        Built with PyTorch · MLflow/DagsHub · AWS S3 · GitHub Actions ·{" "}
        <a
          href="https://github.com/glenlouis8/voltcast"
          target="_blank"
          rel="noreferrer"
          className="text-accent-2 hover:text-accent no-underline"
        >
          source
        </a>
      </div>
    </main>
  );
}
