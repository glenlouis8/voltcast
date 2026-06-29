"use client";

import { useEffect, useState } from "react";
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

  return (
    <main className="container">
      <div className="header">
        <h1>
          <span className="bolt">⚡</span> VoltCast
        </h1>
      </div>
      <p className="subtitle">
        24-hour-ahead US electricity demand forecasts, powered by a from-scratch Transformer.
      </p>

      {/* region tabs */}
      <div className="tabs">
        {REGIONS.map((r) => (
          <button
            key={r}
            className={`tab ${r === region ? "active" : ""}`}
            onClick={() => setRegion(r)}
          >
            {REGION_NAMES[r]}
          </button>
        ))}
      </div>

      {loading && <div className="state">Loading {REGION_NAMES[region]}…</div>}
      {error && <div className="state error">{error}</div>}

      {data && !loading && !error && (
        <>
          {/* champion + stat cards */}
          <div className="cards">
            <div className="card">
              <div className="label">Model</div>
              <div className="value">
                {data.champion.model_type.charAt(0).toUpperCase() +
                  data.champion.model_type.slice(1)}
              </div>
            </div>
            <div className="card">
              <div className="label">Version</div>
              <div className="value">v{data.champion.version}</div>
            </div>
            <div className="card">
              <div className="label">Test MAE</div>
              <div className="value accent">{fmtMW(data.champion.test_mae_mw)}</div>
            </div>
            <div className="card">
              <div className="label">Test WAPE</div>
              <div className="value accent">{fmtPct(data.champion.test_wape)}</div>
            </div>
            <div className="card">
              <div className="label">Peak</div>
              <div className="value">{fmtMW(peak)}</div>
            </div>
            <div className="card">
              <div className="label">Trough</div>
              <div className="value">{fmtMW(trough)}</div>
            </div>
          </div>

          {/* forecast chart */}
          <div className="panel">
            <h2>
              Next 24 hours — {REGION_NAMES[region]}{" "}
              <span className="tz">(UTC)</span>
            </h2>
            <div className="meta">
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
          <div className="panel">
            <h2>
              Hourly forecast <span className="tz">(UTC)</span>
            </h2>
            <div className="table-wrap">
              <table className="forecast-table">
                <thead>
                  <tr>
                    <th>Time</th>
                    <th className="num">Predicted load</th>
                  </tr>
                </thead>
                <tbody>
                  {data.forecast.map((row) => {
                    const d = new Date(row.timestamp);
                    const isPeak = row.predicted_load_mw === peak;
                    const isTrough = row.predicted_load_mw === trough;
                    return (
                      <tr key={row.timestamp}>
                        <td>
                          {d.toLocaleString("en-US", {
                            timeZone: "UTC",
                            weekday: "short",
                            hour: "2-digit",
                            minute: "2-digit",
                            hour12: false,
                          })}
                          {isPeak && <span className="tag peak">peak</span>}
                          {isTrough && <span className="tag trough">trough</span>}
                        </td>
                        <td className="num">{fmtMW(row.predicted_load_mw)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      <div className="footer">
        Built with PyTorch · MLflow/DagsHub · AWS S3 · GitHub Actions ·{" "}
        <a href="https://github.com/glenlouis8/voltcast" target="_blank" rel="noreferrer">
          source
        </a>
      </div>
    </main>
  );
}
