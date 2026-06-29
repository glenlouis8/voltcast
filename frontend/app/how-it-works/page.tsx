import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "How it works — VoltCast",
  description:
    "How VoltCast forecasts US electricity demand: data, model, and the serverless MLOps loop.",
};

const PANEL = "bg-panel border border-border rounded-2xl p-6 mb-5";
const PROSE = "text-text text-[15px] leading-relaxed mb-3 last:mb-0";

// One bulleted item with a small accent dot.
function Bullet({ children }: { children: React.ReactNode }) {
  return (
    <li className="relative pl-[18px] text-text text-[15px] leading-relaxed before:content-[''] before:absolute before:left-0 before:top-[9px] before:w-1.5 before:h-1.5 before:rounded-full before:bg-accent-2">
      {children}
    </li>
  );
}

// Numbered pipeline step.
function Step({ n, children }: { n: number; children: React.ReactNode }) {
  return (
    <li className="flex gap-3.5 items-start text-text text-[15px] leading-relaxed">
      <span className="flex-none w-7 h-7 rounded-lg bg-panel-2 border border-border text-accent text-[13px] font-bold flex items-center justify-center">
        {n}
      </span>
      <div>{children}</div>
    </li>
  );
}

// Static explainer page. No client state — pure server-rendered content, so it
// ships zero JS and loads instantly.
export default function HowItWorks() {
  return (
    <main className="max-w-[1000px] mx-auto px-5 pt-10 pb-20">
      <div className="flex items-baseline gap-3 flex-wrap mb-1.5">
        <h1 className="text-3xl font-bold tracking-tight">
          <span className="text-accent">⚡</span> How VoltCast works
        </h1>
      </div>
      <p className="text-muted text-[15px] mb-7">
        Predicting how much electricity four US grid regions will need over the
        next 24 hours, the same way you'd forecast weather but for power demand.
      </p>

      <Link
        href="/"
        className="inline-block mb-6 text-accent-2 hover:text-accent text-sm font-semibold"
      >
        ← Back to the dashboard
      </Link>

      {/* the one-line pitch */}
      <div className={PANEL}>
        <h2 className="text-base font-semibold mb-2">The problem</h2>
        <p className={PROSE}>
          Every hour, grid operators have to decide how much power to generate.
          Too little means blackouts. Too much wastes money and fuel. VoltCast
          looks at the last week of demand and predicts the next 24 hours for
          California, Texas (ERCOT), the Mid-Atlantic (PJM), and the Midwest
          (MISO).
        </p>
      </div>

      {/* the pipeline, step by step */}
      <div className={PANEL}>
        <h2 className="text-base font-semibold mb-2">The pipeline</h2>
        <p className={PROSE}>
          Data flows one direction: from the government API to the chart you see.
          Each step hands off to the next.
        </p>
        <ol className="list-none mt-4 flex flex-col gap-4">
          <Step n={1}>
            <strong>Pull the data.</strong> The EIA (US Energy Information
            Administration) publishes real hourly demand per region. We pull a
            rolling 5-year window into Parquet files.
          </Step>
          <Step n={2}>
            <strong>Validate it.</strong> A Pandera schema rejects bad rows:
            nulls, negative load, impossible spikes, gaps in time. Corrupt data
            never reaches the model.
          </Step>
          <Step n={3}>
            <strong>Engineer features.</strong> Raw megawatts plus context:
            sin/cos encodings of hour, day, and month (so hour 23 sits next to
            hour 0), lag features (load 1h, 24h, and 168h ago), rolling averages,
            and a weekend flag. Then z-score normalize, scaler fit on the training
            split only — no peeking at the future.
          </Step>
          <Step n={4}>
            <strong>Slide a window.</strong> The long time series becomes thousands
            of training examples: each is 168 hours of input mapped to the 24 hours
            that follow.
          </Step>
          <Step n={5}>
            <strong>Predict.</strong> The champion model takes the latest 168 real
            hours and outputs all 24 future hours in one shot. No feeding
            predictions back in.
          </Step>
          <Step n={6}>
            <strong>Publish.</strong> Forecasts land in S3 as JSON. This dashboard
            reads them through a server route that holds the AWS keys, so the bucket
            stays private and the browser never touches AWS.
          </Step>
        </ol>
      </div>

      {/* the model */}
      <div className={PANEL}>
        <h2 className="text-base font-semibold mb-2">The model</h2>
        <p className={PROSE}>
          A Temporal Transformer, written from scratch in PyTorch. No HuggingFace,
          no Trainer abstractions. It learns which past hours matter most for each
          prediction. To forecast 9pm Friday, it can learn to look hard at 9pm last
          Friday.
        </p>
        <ul className="list-none my-3.5 flex flex-col gap-2.5">
          <Bullet>
            <strong>Input projection</strong> maps the 13 features into a 64-dim
            internal space.
          </Bullet>
          <Bullet>
            <strong>Positional encoding</strong> tells the model where each hour
            sits in the sequence, since attention has no built-in sense of order.
          </Bullet>
          <Bullet>
            <strong>Two attention layers</strong> (4 heads each) weigh the
            relationships between hours.
          </Bullet>
          <Bullet>
            <strong>Output head</strong> turns the last timestep into 24 predicted
            megawatt values.
          </Bullet>
        </ul>
        <p className={PROSE}>
          To prove it earns its place, every region also trains an LSTM and a naive
          "tomorrow = today" baseline. If the Transformer can't beat copy-paste,
          something is broken. It beats the naive baseline by roughly 33%.
        </p>
      </div>

      {/* the MLOps loop */}
      <div className={PANEL}>
        <h2 className="text-base font-semibold mb-2">The serverless loop</h2>
        <p className={PROSE}>
          Nothing runs 24/7. There is no API server. Compute is ephemeral GitHub
          Actions runners, storage is S3 and DagsHub, the dashboard is on Vercel.
          Cost is close to zero.
        </p>
        <ul className="list-none my-3.5 flex flex-col gap-2.5">
          <Bullet>
            <strong>Hourly:</strong> refresh data and regenerate every region's
            24-hour forecast.
          </Bullet>
          <Bullet>
            <strong>Weekly:</strong> check for drift with Evidently. If the data
            distribution moved or the model went stale, retrain. A challenger only
            replaces the champion if it beats it by 1% on the untouched test set.
          </Bullet>
          <Bullet>
            <strong>Tracked:</strong> MLflow on DagsHub logs every run's
            hyperparameters, metrics, and model file, with a champion/challenger
            registry deciding what ships.
          </Bullet>
        </ul>
      </div>

      {/* how to read the numbers */}
      <div className={PANEL}>
        <h2 className="text-base font-semibold mb-2">Reading the numbers</h2>
        <ul className="list-none my-3.5 flex flex-col gap-2.5">
          <Bullet>
            <strong>MAE</strong> (mean absolute error) is the headline metric: on
            average, how many megawatts the forecast is off by. Lower is better.
          </Bullet>
          <Bullet>
            <strong>WAPE</strong> expresses that error as a percentage of total
            demand, so regions of different sizes compare fairly.
          </Bullet>
          <Bullet>
            <strong>All times are UTC.</strong> The dataset is UTC end to end, so
            the chart and table avoid any timezone guessing.
          </Bullet>
        </ul>
      </div>

      <div className="mt-10 text-muted text-[13px] text-center">
        <Link href="/" className="text-accent-2 hover:text-accent no-underline">
          ← Back to the dashboard
        </Link>
        {" · "}
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
