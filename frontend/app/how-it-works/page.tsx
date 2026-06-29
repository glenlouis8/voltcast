import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "How it works — VoltCast",
  description:
    "How VoltCast forecasts US electricity demand: data, model, and the serverless MLOps loop.",
};

// Static explainer page. No client state — pure server-rendered content, so it
// ships zero JS and loads instantly.
export default function HowItWorks() {
  return (
    <main className="container">
      <div className="header">
        <h1>
          <span className="bolt">⚡</span> How VoltCast works
        </h1>
      </div>
      <p className="subtitle">
        Predicting how much electricity four US grid regions will need over the
        next 24 hours, the same way you'd forecast weather but for power demand.
      </p>

      <Link href="/" className="backlink">
        ← Back to the dashboard
      </Link>

      {/* the one-line pitch */}
      <div className="panel">
        <h2>The problem</h2>
        <p className="prose">
          Every hour, grid operators have to decide how much power to generate.
          Too little means blackouts. Too much wastes money and fuel. VoltCast
          looks at the last week of demand and predicts the next 24 hours for
          California, Texas (ERCOT), the Mid-Atlantic (PJM), and the Midwest
          (MISO).
        </p>
      </div>

      {/* the pipeline, step by step */}
      <div className="panel">
        <h2>The pipeline</h2>
        <p className="prose">
          Data flows one direction: from the government API to the chart you see.
          Each step hands off to the next.
        </p>
        <ol className="steps">
          <li>
            <span className="step-n">1</span>
            <div>
              <strong>Pull the data.</strong> The EIA (US Energy Information
              Administration) publishes real hourly demand per region. We pull a
              rolling 5-year window into Parquet files.
            </div>
          </li>
          <li>
            <span className="step-n">2</span>
            <div>
              <strong>Validate it.</strong> A Pandera schema rejects bad rows:
              nulls, negative load, impossible spikes, gaps in time. Corrupt data
              never reaches the model.
            </div>
          </li>
          <li>
            <span className="step-n">3</span>
            <div>
              <strong>Engineer features.</strong> Raw megawatts plus context:
              sin/cos encodings of hour, day, and month (so hour 23 sits next to
              hour 0), lag features (load 1h, 24h, and 168h ago), rolling
              averages, and a weekend flag. Then z-score normalize, scaler fit on
              the training split only — no peeking at the future.
            </div>
          </li>
          <li>
            <span className="step-n">4</span>
            <div>
              <strong>Slide a window.</strong> The long time series becomes
              thousands of training examples: each is 168 hours of input mapped to
              the 24 hours that follow.
            </div>
          </li>
          <li>
            <span className="step-n">5</span>
            <div>
              <strong>Predict.</strong> The champion model takes the latest 168
              real hours and outputs all 24 future hours in one shot. No feeding
              predictions back in.
            </div>
          </li>
          <li>
            <span className="step-n">6</span>
            <div>
              <strong>Publish.</strong> Forecasts land in S3 as JSON. This
              dashboard reads them through a server route that holds the AWS keys,
              so the bucket stays private and the browser never touches AWS.
            </div>
          </li>
        </ol>
      </div>

      {/* the model */}
      <div className="panel">
        <h2>The model</h2>
        <p className="prose">
          A Temporal Transformer, written from scratch in PyTorch. No HuggingFace,
          no Trainer abstractions. It learns which past hours matter most for each
          prediction. To forecast 9pm Friday, it can learn to look hard at 9pm
          last Friday.
        </p>
        <ul className="plain-list">
          <li>
            <strong>Input projection</strong> maps the 13 features into a 64-dim
            internal space.
          </li>
          <li>
            <strong>Positional encoding</strong> tells the model where each hour
            sits in the sequence, since attention has no built-in sense of order.
          </li>
          <li>
            <strong>Two attention layers</strong> (4 heads each) weigh the
            relationships between hours.
          </li>
          <li>
            <strong>Output head</strong> turns the last timestep into 24 predicted
            megawatt values.
          </li>
        </ul>
        <p className="prose">
          To prove it earns its place, every region also trains an LSTM and a
          naive "tomorrow = today" baseline. If the Transformer can't beat copy-
          paste, something is broken. It beats the naive baseline by roughly 33%.
        </p>
      </div>

      {/* the MLOps loop */}
      <div className="panel">
        <h2>The serverless loop</h2>
        <p className="prose">
          Nothing runs 24/7. There is no API server. Compute is ephemeral GitHub
          Actions runners, storage is S3 and DagsHub, the dashboard is on Vercel.
          Cost is close to zero.
        </p>
        <ul className="plain-list">
          <li>
            <strong>Hourly:</strong> refresh data and regenerate every region's
            24-hour forecast.
          </li>
          <li>
            <strong>Weekly:</strong> check for drift with Evidently. If the data
            distribution moved or the model went stale, retrain. A challenger only
            replaces the champion if it beats it by 1% on the untouched test set.
          </li>
          <li>
            <strong>Tracked:</strong> MLflow on DagsHub logs every run's
            hyperparameters, metrics, and model file, with a champion/challenger
            registry deciding what ships.
          </li>
        </ul>
      </div>

      {/* how to read the numbers */}
      <div className="panel">
        <h2>Reading the numbers</h2>
        <ul className="plain-list">
          <li>
            <strong>MAE</strong> (mean absolute error) is the headline metric: on
            average, how many megawatts the forecast is off by. Lower is better.
          </li>
          <li>
            <strong>WAPE</strong> expresses that error as a percentage of total
            demand, so regions of different sizes compare fairly.
          </li>
          <li>
            <strong>All times are UTC.</strong> The dataset is UTC end to end, so
            the chart and table avoid any timezone guessing.
          </li>
        </ul>
      </div>

      <div className="footer">
        <Link href="/" className="footer-link">
          ← Back to the dashboard
        </Link>
        {" · "}
        <a
          href="https://github.com/glenlouis8/voltcast"
          target="_blank"
          rel="noreferrer"
        >
          source
        </a>
      </div>
    </main>
  );
}
