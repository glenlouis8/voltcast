// frontend/app/api/forecast/[region]/route.ts
//
// The "middleman". This is server-side code — it runs on Vercel's server, NOT
// in the browser. The browser only ever sees the JSON it returns, never the AWS
// keys used to fetch it. That's what keeps the S3 bucket private + safe.
//
// Browser → GET /api/forecast/CAL → (this function reads private S3) → JSON

import { S3Client, GetObjectCommand } from "@aws-sdk/client-s3";

// Only these regions are allowed — never trust the URL blindly (stops someone
// asking for /api/forecast/../../secret).
const REGIONS = new Set(["CAL", "TEX", "PJM", "MISO"]);

// One S3 client, reused across requests. Credentials come from server-side env
// vars (set in Vercel, NOT prefixed NEXT_PUBLIC, so they never reach the browser).
const s3 = new S3Client({
  region: process.env.AWS_DEFAULT_REGION,
  credentials: {
    accessKeyId: process.env.AWS_ACCESS_KEY_ID!,
    secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY!,
  },
});

export async function GET(
  _req: Request,
  { params }: { params: { region: string } }
) {
  const region = params.region.toUpperCase();

  if (!REGIONS.has(region)) {
    return Response.json({ error: "unknown region" }, { status: 404 });
  }

  try {
    const out = await s3.send(
      new GetObjectCommand({
        Bucket: process.env.S3_BUCKET,
        Key: `forecasts/${region}.json`,
      })
    );
    // The object body is a stream; transformToString() reads it fully.
    const body = await out.Body!.transformToString();

    // Return the JSON straight through. Cache 5 min at the edge so we don't hit
    // S3 on every page load (forecasts only change hourly anyway).
    return new Response(body, {
      headers: {
        "content-type": "application/json",
        "cache-control": "public, s-maxage=300, stale-while-revalidate=600",
      },
    });
  } catch {
    return Response.json(
      { error: `no forecast for ${region} yet` },
      { status: 404 }
    );
  }
}
