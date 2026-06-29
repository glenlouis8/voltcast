import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "VoltCast — US Electricity Demand Forecasting",
  description:
    "24-hour-ahead electricity demand forecasts for US grid regions, powered by a from-scratch Transformer.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
