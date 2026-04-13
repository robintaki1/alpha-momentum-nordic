(() => {
  function parseCsv(text) {
    const lines = text.replace(/\r/g, "").split("\n").filter((line) => line.trim().length > 0);
    if (lines.length < 2) {
      return [];
    }

    const headers = lines[0].split(",").map((header) => header.trim());
    return lines.slice(1).map((line) => {
      const values = line.split(",").map((value) => value.trim());
      const row = {};
      headers.forEach((header, index) => {
        row[header] = values[index] ?? "";
      });
      return row;
    });
  }

  function formatCurrency(value) {
    const numericValue = Number(value);
    if (!Number.isFinite(numericValue)) {
      return value || "n/a";
    }
    return `${numericValue.toLocaleString("en-US", {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    })} SEK`;
  }

  function cleanText(value) {
    const text = value == null ? "" : String(value).trim();
    return text.length ? text : "n/a";
  }

  function renderLedgerCard(row) {
    const article = document.createElement("article");
    article.className = "snapshot-card";

    const label = document.createElement("span");
    label.className = "snapshot-label";
    label.textContent = cleanText(row.trade_date);

    const value = document.createElement("strong");
    value.className = "snapshot-value";
    value.textContent = formatCurrency(row.portfolio_value_after_sek);

    const note = document.createElement("p");
    note.className = "snapshot-note";
    note.textContent = [
      `Holding month ${cleanText(row.holding_month)}.`,
      `Before ${formatCurrency(row.portfolio_value_before_sek)}.`,
      `After costs ${formatCurrency(row.portfolio_value_after_sek)}.`,
      `Total cost ${formatCurrency(row.total_cost_sek)}.`,
      `Cash ${formatCurrency(row.cash_sek)}.`,
    ].join(" ");

    article.append(label, value, note);
    return article;
  }

  async function hydrateLedgerSection(section) {
    const source = section.dataset.historySource;
    const grid = section.querySelector("[data-history-grid]");
    if (!source || !grid) {
      return;
    }

    try {
      const response = await fetch(source, { cache: "no-store" });
      if (!response.ok) {
        return;
      }

      const rows = parseCsv(await response.text());
      if (!rows.length) {
        return;
      }

      grid.replaceChildren(...rows.map(renderLedgerCard));
    } catch {
      // Keep the static fallback cards if the CSV is unavailable.
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll("[data-history-source]").forEach((section) => {
      void hydrateLedgerSection(section);
    });
  });
})();
