import React from "react";
import { Link } from "react-router-dom";

const HomePage = () => {
  return (
    <div className="min-h-screen px-4 py-6 md:px-6 md:py-8">
      <div className="mx-auto max-w-6xl space-y-4">
        <header className="terminal-card rounded-xl p-5 md:p-6">
          <p className="font-mono text-xs uppercase tracking-[0.22em] text-terminal-muted">Smart Stock Platform</p>
          <h1 className="mt-2 font-display text-3xl text-terminal-text md:text-5xl">Market Terminal + Forecast Studio</h1>
          <p className="mt-3 max-w-3xl text-sm text-terminal-muted md:text-base">
            Unified dashboard for market microstructure signals, sentiment flow, advanced analytics, and model-ready projections.
            Terminal is now the main experience.
          </p>
          <div className="mt-5 flex flex-wrap gap-3">
            <Link
              to="/"
              className="rounded-md border border-terminal-accent bg-terminal-accent/20 px-4 py-2 font-mono text-xs uppercase tracking-[0.12em] text-terminal-text transition hover:bg-terminal-accent/30"
            >
              Open Terminal
            </Link>
            <Link
              to="/model-comparison"
              className="rounded-md border border-terminal-border bg-terminal-panelSoft px-4 py-2 font-mono text-xs uppercase tracking-[0.12em] text-terminal-text transition hover:border-terminal-accent/60"
            >
              Model Lab
            </Link>
          </div>
        </header>

        <section className="grid grid-cols-1 gap-4 md:grid-cols-3">
          <div className="terminal-card rounded-xl p-4">
            <h2 className="font-display text-lg text-terminal-text">Terminal First</h2>
            <p className="mt-2 text-sm text-terminal-muted">
              Watchlist, movers, sector breadth, index tracker, sentiment tape, and volatility/returns panels in one dense workspace.
            </p>
          </div>
          <div className="terminal-card rounded-xl p-4">
            <h2 className="font-display text-lg text-terminal-text">Forecast Panels</h2>
            <p className="mt-2 text-sm text-terminal-muted">
              Projection charts are integrated directly in terminal with horizon/model controls, even when models are in placeholder mode.
            </p>
          </div>
          <div className="terminal-card rounded-xl p-4">
            <h2 className="font-display text-lg text-terminal-text">Research Ready</h2>
            <p className="mt-2 text-sm text-terminal-muted">
              Structured API outputs and visual diagnostics for presenting analytics quality before final model pipeline stabilization.
            </p>
          </div>
        </section>
      </div>
    </div>
  );
};

export default HomePage;
