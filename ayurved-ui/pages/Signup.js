import { useState, useMemo, useCallback, useRef, useEffect } from "react";

// ─── Stopwords & NLP helpers (client-side TF-IDF search) ──────────────────
const STOP = new Set(["a","an","the","and","or","but","in","on","at","to","for","of","with","by","from","is","are","was","were","be","been","have","has","had","do","does","did","will","would","could","should","may","might","can","not","no","it","its","as","if","that","this","these","those","also","very","just","only","all","any","i","me","my","we","you","your","he","she","they","their","them","what","there","so","too","even","both","which","who","how","when","where","why"]);

function tokenize(text) {
  if (!text) return [];
  return text.toLowerCase().replace(/[^a-z\s]/g, " ").split(/\s+/).filter(t => t.length > 2 && !STOP.has(t));
}

function stemLight(w) {
  if (w.length <= 4) return w;
  if (w.endsWith("ing")) return w.slice(0, -3);
  if (w.endsWith("tion")) return w.slice(0, -4);
  if (w.endsWith("ness")) return w.slice(0, -4);
  if (w.endsWith("ies")) return w.slice(0, -3) + "y";
  if (w.endsWith("ed")) return w.slice(0, -2);
  if (w.endsWith("er")) return w.slice(0, -2);
  if (w.endsWith("s") && !w.endsWith("ss")) return w.slice(0, -1);
  return w;
}

function processQuery(q) {
  return tokenize(q).map(stemLight);
}

function scoreRecord(record, queryTokens) {
  if (!queryTokens.length) return 0;
  const symTokens = tokenize(record.symptoms).map(stemLight);
  const descTokens = tokenize(record.description).map(stemLight);
  const nameTokens = tokenize(record.formulation_name).map(stemLight);
  const ingTokens = tokenize(record.ingredients).map(stemLight);
  const allTokens = [...symTokens, ...symTokens, ...descTokens, ...nameTokens, ...ingTokens];
  const tokenSet = new Set(allTokens);
  let hits = 0;
  queryTokens.forEach(qt => {
    if (tokenSet.has(qt)) hits++;
    else allTokens.forEach(t => { if (t.includes(qt) || qt.includes(t)) hits += 0.5; });
  });
  return hits / queryTokens.length;
}

// ─── Dataset (embedded) ────────────────────────────────────────────────────

// ─── Category config ───────────────────────────────────────────────────────
const CAT_COLORS = {
  rasa:     { bg: "#FFF0E6", text: "#8B3A0F", border: "#F4A56A" },
  taila:    { bg: "#E6F7F0", text: "#0F5C3A", border: "#6ABFA0" },
  vati:     { bg: "#EEF0FF", text: "#2D35A0", border: "#8B96F4" },
  churna:   { bg: "#FFF8E6", text: "#8B5E0F", border: "#F4C96A" },
  ghrita:   { bg: "#FFF0F5", text: "#8B0F45", border: "#F46A95" },
  kashayam: { bg: "#E6F0FF", text: "#0F3A8B", border: "#6A9BF4" },
  avaleha:  { bg: "#F0FFE6", text: "#2A7A10", border: "#7AD45A" },
  arishta:  { bg: "#F5EEF8", text: "#5A0E85", border: "#BA8FD4" },
  bhasma:   { bg: "#F0F0F0", text: "#404040", border: "#A0A0A0" },
  asava:    { bg: "#FFF4E6", text: "#7A3A0A", border: "#D4996A" },
  ark:      { bg: "#E6FFFE", text: "#0A5E5A", border: "#6AD4CF" },
  guggulu:  { bg: "#FFE6EE", text: "#7A0A2E", border: "#D46A8E" },
};

function getCatStyle(cat) {
  return CAT_COLORS[cat?.toLowerCase()] || { bg: "#F5F5F5", text: "#555", border: "#CCC" };
}

const CATEGORY_ICONS = {
  rasa: "⚗️", taila: "🫙", vati: "💊", churna: "🌿", ghrita: "🧈",
  kashayam: "🍵", avaleha: "🍯", arishta: "🫗", bhasma: "🔬",
  asava: "🫧", ark: "💧", guggulu: "🌱", default: "🌺"
};

function getCatIcon(cat) {
  return CATEGORY_ICONS[cat?.toLowerCase()] || CATEGORY_ICONS.default;
}

// ─── Main App ──────────────────────────────────────────────────────────────
export default function Recommend() {
  const [query, setQuery] = useState("");
  const [activeQuery, setActiveQuery] = useState("");
  const [selectedCat, setSelectedCat] = useState("all");
  const [apiResults, setApiResults] = useState([]);
  const [selectedDoc, setSelectedDoc] = useState("all");
  const [selectedCard, setSelectedCard] = useState(null);
  const [page, setPage] = useState(1);
  const [sortBy, setSortBy] = useState("relevance");
  const [view, setView] = useState("grid"); // grid | list
  const inputRef = useRef(null);
  const PER_PAGE = 5;

  // Parse embedded data
  const DATA = apiResults;

  // All categories & doctor types
  const allCats = useMemo(() => {
    const s = new Set(DATA.map(d => d.category).filter(Boolean));
    return ["all", ...Array.from(s).sort()];
  }, [DATA]);

  const allDocs = useMemo(() => {
    const s = new Set();
    DATA.forEach(d => {
      if (d.doctor_type) d.doctor_type.split(/[\s,\/]+/).filter(Boolean).forEach(t => s.add(t.toLowerCase()));
    });
    return ["all", ...Array.from(s).sort()];
  }, [DATA]);

  // Scored & filtered results
  const results = useMemo(() => {
    const queryTokens = processQuery(activeQuery);
    let scored = DATA.map(r => ({
      ...r,
      _score: activeQuery ? scoreRecord(r, queryTokens) : 1,
    }));

    if (activeQuery) scored = scored.filter(r => r._score > 0);
    if (selectedCat !== "all") scored = scored.filter(r => r.category?.toLowerCase() === selectedCat);
    if (selectedDoc !== "all") scored = scored.filter(r => r.doctor_type?.toLowerCase().includes(selectedDoc));

    if (sortBy === "relevance") scored.sort((a, b) => b._score - a._score);
    else if (sortBy === "name") scored.sort((a, b) => a.formulation_name.localeCompare(b.formulation_name));
    else if (sortBy === "category") scored.sort((a, b) => (a.category || "").localeCompare(b.category || ""));

    return scored;
  }, [DATA, activeQuery, selectedCat, selectedDoc, sortBy]);

  const totalPages = Math.ceil(results.length / PER_PAGE);
  const pageResults = results.slice((page - 1) * PER_PAGE, page * PER_PAGE);

  async function handleSearch() {
  if (!query) return;

  try {
    const res = await fetch("http://127.0.0.1:5050/recommend", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ query, top_n: 5 }),
    });

    const data = await res.json();
    setApiResults(data.results || []);
    setActiveQuery(query);
    setPage(1);
    setSelectedCard(null);
  } catch (err) {
    console.error(err);
  }
}

  function handleKeyDown(e) {
    if (e.key === "Enter") handleSearch();
  }

  function handlePill(q) {
    setQuery(q);
    setActiveQuery(q);
    setPage(1);
    setSelectedCard(null);
  }

  function clearSearch() {
    setQuery("");
    setActiveQuery("");
    setPage(1);
    setSelectedCard(null);
    inputRef.current?.focus();
  }

  const PILLS = [
    "bloating nausea stomach pain",
    "stress anxiety insomnia",
    "joint pain arthritis",
    "cough respiratory cold",
    "skin rash eczema",
    "hair loss dandruff",
    "diabetes blood sugar",
    "fever headache weakness",
  ];

  return (
    <div style={{ fontFamily: "'Cormorant Garamond', Georgia, serif", minHeight: "100vh", background: "#FDFAF5", color: "#2C1810" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;0,600;1,300;1,400&family=Nunito:wght@300;400;600;700&display=swap');
        * { box-sizing: border-box; }
        body { margin: 0; }
        ::placeholder { color: #B0967A; opacity: 1; }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: #F5EFE6; }
        ::-webkit-scrollbar-thumb { background: #C4A882; border-radius: 3px; }
        .card-hover { transition: transform 0.2s ease, box-shadow 0.2s ease; }
        .card-hover:hover { transform: translateY(-3px); box-shadow: 0 12px 32px rgba(139,90,43,0.15) !important; }
        .pill-btn { transition: all 0.15s ease; cursor: pointer; }
        .pill-btn:hover { background: #8B5C2A !important; color: #FDF8F0 !important; }
        .search-btn { transition: all 0.15s ease; }
        .search-btn:hover { background: #6B3D1A !important; }
        .filter-chip { transition: all 0.15s ease; cursor: pointer; }
        .filter-chip:hover { opacity: 0.85; }
        .page-btn { transition: all 0.15s ease; cursor: pointer; }
        .page-btn:hover { background: #8B5C2A !important; color: #FFF !important; }
        .close-btn:hover { background: rgba(0,0,0,0.08) !important; }
        .step-item { counter-increment: steps; }
        .step-item::before { content: counter(steps); display: inline-flex; width: 22px; height: 22px; background: #8B5C2A; color: #FFF; border-radius: 50%; font-size: 11px; align-items: center; justify-content: center; font-family: 'Nunito', sans-serif; flex-shrink: 0; margin-right: 10px; margin-top: 2px; }
        .steps-list { counter-reset: steps; list-style: none; padding: 0; margin: 0; }
        .tag-doctor { background: #E8F4F8; color: #1A5F7A; border: 1px solid #A8D8EA; border-radius: 99px; padding: 3px 10px; font-size: 11px; font-family: 'Nunito', sans-serif; font-weight: 600; white-space: nowrap; }
        @keyframes fadeIn { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:translateY(0); } }
        .fade-in { animation: fadeIn 0.3s ease forwards; }
        @keyframes slideUp { from { opacity:0; transform:translateY(40px); } to { opacity:1; transform:translateY(0); } }
        .slide-up { animation: slideUp 0.35s ease forwards; }
      `}</style>

      {/* Header */}
      <div style={{ background: "linear-gradient(135deg, #3D1C08 0%, #6B3D1A 50%, #8B5C2A 100%)", padding: "2.5rem 2rem 2rem", textAlign: "center" }}>
        <div style={{ maxWidth: 860, margin: "0 auto" }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 10, marginBottom: 8 }}>
            <span style={{ fontSize: 28 }}>🌿</span>
            <h1 style={{ margin: 0, fontSize: 34, fontWeight: 400, color: "#F5DEB3", letterSpacing: 1, fontStyle: "italic" }}>
              Ayurveda Formulation Recommendation System 
            </h1>
            <span style={{ fontSize: 28 }}>🌿</span>
          </div>
          <p style={{ margin: "0 0 1.5rem", color: "#C4A882", fontSize: 14, fontFamily: "'Nunito', sans-serif", fontWeight: 300, letterSpacing: 0.5 }}>
            {DATA.length} classical formulations · Search by symptom, ingredient, or condition
          </p>

          {/* Search bar */}
          <div style={{ display: "flex", gap: 8, maxWidth: 680, margin: "0 auto 1rem" }}>
            <div style={{ flex: 1, position: "relative" }}>
              <input
                ref={inputRef}
                value={query}
                onChange={e => setQuery(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="e.g. bloating, nausea, joint pain..."
                style={{ width: "100%", padding: "13px 40px 13px 16px", fontSize: 15, fontFamily: "'Cormorant Garamond', serif", border: "1.5px solid #C4A882", borderRadius: 12, background: "#FDFAF5", color: "#2C1810", outline: "none" }}
              />
              {query && (
                <button onClick={clearSearch} style={{ position: "absolute", right: 10, top: "50%", transform: "translateY(-50%)", background: "none", border: "none", cursor: "pointer", color: "#B0967A", fontSize: 18, padding: 2 }}>×</button>
              )}
            </div>
            <button className="search-btn" onClick={handleSearch} style={{ padding: "13px 24px", background: "#8B5C2A", color: "#FDF8F0", border: "none", borderRadius: 12, fontSize: 14, fontFamily: "'Nunito', sans-serif", fontWeight: 700, cursor: "pointer", whiteSpace: "nowrap" }}>
              Search →
            </button>
          </div>

          {/* Pills */}
          <div style={{ display: "flex", flexWrap: "wrap", gap: 6, justifyContent: "center" }}>
            {PILLS.map(p => (
              <button key={p} className="pill-btn" onClick={() => handlePill(p)}
                style={{ padding: "5px 14px", background: "rgba(245,222,179,0.15)", color: "#F5DEB3", border: "1px solid rgba(196,168,130,0.4)", borderRadius: 99, fontSize: 12, fontFamily: "'Nunito', sans-serif", cursor: "pointer" }}>
                {p}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div style={{ maxWidth: 1200, margin: "0 auto", padding: "1.5rem 1.5rem" }}>

        {/* Filters row */}
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginBottom: 16, alignItems: "center" }}>
          <span style={{ fontSize: 12, fontFamily: "'Nunito', sans-serif", color: "#8B6A4A", fontWeight: 600, marginRight: 4 }}>FILTER:</span>

          {/* Category filter */}
          <select value={selectedCat} onChange={e => { setSelectedCat(e.target.value); setPage(1); }}
            style={{ padding: "6px 12px", border: "1px solid #C4A882", borderRadius: 8, fontSize: 13, fontFamily: "'Nunito', sans-serif", background: "#FDFAF5", color: "#2C1810", cursor: "pointer" }}>
            {allCats.map(c => <option key={c} value={c}>{c === "all" ? "All Categories" : c}</option>)}
          </select>

          {/* Doctor filter */}
          <select value={selectedDoc} onChange={e => { setSelectedDoc(e.target.value); setPage(1); }}
            style={{ padding: "6px 12px", border: "1px solid #C4A882", borderRadius: 8, fontSize: 13, fontFamily: "'Nunito', sans-serif", background: "#FDFAF5", color: "#2C1810", cursor: "pointer" }}>
            {allDocs.map(d => <option key={d} value={d}>{d === "all" ? "All Specialists" : d}</option>)}
          </select>

          {/* Sort */}
          <select value={sortBy} onChange={e => { setSortBy(e.target.value); setPage(1); }}
            style={{ padding: "6px 12px", border: "1px solid #C4A882", borderRadius: 8, fontSize: 13, fontFamily: "'Nunito', sans-serif", background: "#FDFAF5", color: "#2C1810", cursor: "pointer" }}>
            <option value="relevance">Sort: Relevance</option>
            <option value="name">Sort: A–Z Name</option>
            <option value="category">Sort: Category</option>
          </select>

          {/* View toggle */}
          <div style={{ marginLeft: "auto", display: "flex", gap: 4 }}>
            {["grid", "list"].map(v => (
              <button key={v} onClick={() => setView(v)}
                style={{ padding: "6px 12px", border: "1px solid #C4A882", borderRadius: 8, fontSize: 12, fontFamily: "'Nunito', sans-serif", background: view === v ? "#8B5C2A" : "#FDFAF5", color: view === v ? "#FFF" : "#8B5C2A", cursor: "pointer", fontWeight: 600 }}>
                {v === "grid" ? "⊞ Grid" : "≡ List"}
              </button>
            ))}
          </div>
        </div>

        {/* Results count */}
        <div style={{ marginBottom: 16, fontFamily: "'Nunito', sans-serif", fontSize: 13, color: "#8B6A4A" }}>
          {activeQuery
            ? <span>Found <strong>{results.length}</strong> formulations for "<em>{activeQuery}</em>"</span>
            : <span>Showing <strong>{results.length}</strong> formulations</span>
          }
          {(selectedCat !== "all" || selectedDoc !== "all") && (
            <span> · <button onClick={() => { setSelectedCat("all"); setSelectedDoc("all"); }} style={{ background: "none", border: "none", color: "#8B5C2A", cursor: "pointer", fontSize: 13, fontFamily: "'Nunito', sans-serif", textDecoration: "underline" }}>Clear filters</button></span>
          )}
        </div>

        {/* Grid / List */}
        {results.length === 0 ? (
          <div style={{ textAlign: "center", padding: "4rem", color: "#B0967A", fontStyle: "italic", fontSize: 18 }}>
            No formulations found. Try a different query or clear filters.
          </div>
        ) : view === "grid" ? (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))", gap: 16, marginBottom: 24 }}>
            {pageResults.map((r, i) => (
              <GridCard key={r.id} r={r} i={i} onClick={() => setSelectedCard(r)} />
            ))}
          </div>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: 10, marginBottom: 24 }}>
            {pageResults.map((r, i) => (
              <ListCard key={r.id} r={r} i={i} onClick={() => setSelectedCard(r)} />
            ))}
          </div>
        )}

        {/* Pagination */}
        {totalPages > 1 && (
          <div style={{ display: "flex", gap: 6, justifyContent: "center", flexWrap: "wrap", marginTop: 8 }}>
            <button className="page-btn" onClick={() => setPage(p => Math.max(1, p - 1))} disabled={page === 1}
              style={{ padding: "7px 14px", border: "1px solid #C4A882", borderRadius: 8, background: "#FDFAF5", color: page === 1 ? "#C4A882" : "#8B5C2A", cursor: page === 1 ? "default" : "pointer", fontFamily: "'Nunito', sans-serif", fontSize: 13 }}>
              ← Prev
            </button>
            {Array.from({ length: Math.min(7, totalPages) }, (_, i) => {
              let p;
              if (totalPages <= 7) p = i + 1;
              else if (page <= 4) p = i + 1;
              else if (page >= totalPages - 3) p = totalPages - 6 + i;
              else p = page - 3 + i;
              return (
                <button key={p} className="page-btn" onClick={() => setPage(p)}
                  style={{ padding: "7px 13px", border: "1px solid #C4A882", borderRadius: 8, background: page === p ? "#8B5C2A" : "#FDFAF5", color: page === p ? "#FFF" : "#8B5C2A", fontFamily: "'Nunito', sans-serif", fontSize: 13, fontWeight: page === p ? 700 : 400 }}>
                  {p}
                </button>
              );
            })}
            <button className="page-btn" onClick={() => setPage(p => Math.min(totalPages, p + 1))} disabled={page === totalPages}
              style={{ padding: "7px 14px", border: "1px solid #C4A882", borderRadius: 8, background: "#FDFAF5", color: page === totalPages ? "#C4A882" : "#8B5C2A", cursor: page === totalPages ? "default" : "pointer", fontFamily: "'Nunito', sans-serif", fontSize: 13 }}>
              Next →
            </button>
          </div>
        )}
      </div>

      {/* Detail Modal */}
      {selectedCard && (
        <DetailModal r={selectedCard} onClose={() => setSelectedCard(null)} />
      )}
    </div>
  );
}

// ─── Grid Card ────────────────────────────────────────────────────────────
function GridCard({ r, i, onClick }) {
  const cs = getCatStyle(r.category);
  const icon = getCatIcon(r.category);
  const doctors = r.doctor_type ? r.doctor_type.split(/[\s,\/]+/).filter(Boolean).slice(0, 2) : [];

  return (
    <div className="card-hover fade-in" onClick={onClick}
      style={{ background: "#FFF", border: "1px solid #E8D9C5", borderRadius: 14, padding: "1.1rem 1.2rem", cursor: "pointer", animationDelay: `${i * 0.03}s`, display: "flex", flexDirection: "column", gap: 10 }}>

              {/* Top row */}
        <div style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "flex-start"
        }}>

          {/* LEFT SIDE (icon + title) */}
          <div style={{ display: "flex", alignItems: "flex-start", gap: 10 }}>
            <div style={{
              width: 40,
              height: 40,
              background: cs.bg,
              border: `1px solid ${cs.border}`,
              borderRadius: 10,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: 18,
              flexShrink: 0
            }}>
              {icon}
            </div>

            <div>
              <h3 style={{
                margin: 0,
                fontSize: 15.5,
                fontWeight: 500,
                color: "#2C1810",
                lineHeight: 1.3,
                fontFamily: "'Cormorant Garamond', serif"
              }}>
                {r.formulation_name}
              </h3>

              {r.category && (
                <span style={{
                  fontSize: 10,
                  fontFamily: "'Nunito', sans-serif",
                  fontWeight: 700,
                  textTransform: "uppercase",
                  letterSpacing: 0.8,
                  color: cs.text,
                  background: cs.bg,
                  padding: "2px 8px",
                  borderRadius: 99,
                  border: `1px solid ${cs.border}`
                }}>
                  {r.category}
                </span>
              )}
            </div>
          </div>

          {/* RIGHT SIDE (DOSHA) */}
          {r.dosha && (
            <div style={{ display: "flex", gap: 4 }}>
              {r.dosha.map((d, i) => (
                <span
                  key={i}
                  style={{
                    fontSize: 10,
                    padding: "3px 8px",
                    borderRadius: 999,
                    fontFamily: "'Nunito', sans-serif",
                    fontWeight: 600,

                    background:
                      d === "Vata" ? "#E8F1FF" :
                      d === "Pitta" ? "#FFEAEA" :
                      d === "Kapha" ? "#E9F7EF" : "#F5EFE6",

                    color:
                      d === "Vata" ? "#3A6FD8" :
                      d === "Pitta" ? "#C0392B" :
                      d === "Kapha" ? "#1E8449" : "#8B5C2A",

                    border:
                      d === "Vata" ? "1px solid #B7D0FF" :
                      d === "Pitta" ? "1px solid #F5B7B1" :
                      d === "Kapha" ? "1px solid #ABEBC6" : "1px solid #E0D6C8",
                  }}
                >
                  {d}
                </span>
              ))}
            </div>
          )}

        </div>

      {/* Description */}
      <p style={{ margin: 0, fontSize: 12.5, color: "#6B4A2A", lineHeight: 1.6, fontStyle: "italic", display: "-webkit-box", WebkitLineClamp: 3, WebkitBoxOrient: "vertical", overflow: "hidden" }}>
        {r.description || "Classical Ayurvedic formulation."}
      </p>

      {/* Symptoms tags */}
      {r.symptoms && (
        <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
          {r.symptoms.split(" ").slice(0, 5).map((s, j) => (
            <span key={j} style={{ fontSize: 10, fontFamily: "'Nunito', sans-serif", background: "#F5EFE6", color: "#8B5C2A", padding: "2px 7px", borderRadius: 99, border: "1px solid #E8D9C5" }}>
              {s}
            </span>
          ))}
          {r.symptoms.split(" ").length > 5 && (
            <span style={{ fontSize: 10, fontFamily: "'Nunito', sans-serif", color: "#B0967A" }}>+{r.symptoms.split(" ").length - 5}</span>
          )}
        </div>
      )}

      {/* Footer */}
      <div style={{ marginTop: "auto", display: "flex", alignItems: "center", justifyContent: "space-between", paddingTop: 8, borderTop: "1px solid #F0E6D6" }}>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
          {doctors.map(d => <span key={d} className="tag-doctor">{d}</span>)}
        </div>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          
          {r.product_link && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                window.open(r.product_link, "_blank");
              }}
              style={{
                fontSize: 11,
                padding: "4px 10px",
                background: "#8B5C2A",
                color: "#FFF",
                border: "none",
                borderRadius: 6,
                cursor: "pointer"
              }}
            >
              Buy
            </button>
          )}

          <span style={{ fontSize: 11, fontFamily: "'Nunito', sans-serif", color: "#8B5C2A", fontWeight: 700 }}>
            View →
          </span>

        </div>      </div>
            </div>
          );
}

// ─── List Card ────────────────────────────────────────────────────────────
function ListCard({ r, i, onClick }) {
  const cs = getCatStyle(r.category);
  const icon = getCatIcon(r.category);
  const doctors = r.doctor_type ? r.doctor_type.split(/[\s,\/]+/).filter(Boolean).slice(0, 3) : [];

  return (
    <div className="card-hover fade-in" onClick={onClick}
      style={{ background: "#FFF", border: "1px solid #E8D9C5", borderRadius: 12, padding: "1rem 1.25rem", cursor: "pointer", display: "flex", gap: 14, alignItems: "flex-start", animationDelay: `${i * 0.02}s` }}>

      <div style={{ width: 38, height: 38, background: cs.bg, border: `1px solid ${cs.border}`, borderRadius: 10, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 17, flexShrink: 0 }}>
        {icon}
      </div>

      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ display: "flex", alignItems: "baseline", gap: 8, flexWrap: "wrap", marginBottom: 4 }}>
          <h3 style={{ margin: 0, fontSize: 16, fontWeight: 500, color: "#2C1810", fontFamily: "'Cormorant Garamond', serif" }}>
            {r.formulation_name}
          </h3>
          {r.category && (
            <span style={{ fontSize: 10, fontFamily: "'Nunito', sans-serif", fontWeight: 700, textTransform: "uppercase", letterSpacing: 0.8, color: cs.text, background: cs.bg, padding: "2px 8px", borderRadius: 99 }}>
              {r.category}
            </span>
          )}
        </div>
        <p style={{ margin: "0 0 6px", fontSize: 13, color: "#6B4A2A", lineHeight: 1.5, fontStyle: "italic", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
          {r.description}
        </p>
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap", alignItems: "center" }}>
          {r.symptoms.split(" ").slice(0, 6).map((s, j) => (
            <span key={j} style={{ fontSize: 10, fontFamily: "'Nunito', sans-serif", background: "#F5EFE6", color: "#8B5C2A", padding: "2px 7px", borderRadius: 99, border: "1px solid #E8D9C5" }}>
              {s}
            </span>
          ))}
        </div>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 5, alignItems: "flex-end", flexShrink: 0 }}>
        {doctors.map(d => <span key={d} className="tag-doctor">{d}</span>)}
        <span style={{ fontSize: 11, fontFamily: "'Nunito', sans-serif", color: "#8B5C2A", fontWeight: 700, marginTop: 4 }}>View →</span>
      </div>
    </div>
  );
}

// ─── Detail Modal ─────────────────────────────────────────────────────────
function DetailModal({ r, onClose }) {
  const cs = getCatStyle(r.category);
  const icon = getCatIcon(r.category);
  const steps = r.steps ? r.steps.split(/(?<=[.!?])\s+|(?<=\d)\s+(?=[a-z])/).filter(s => s.trim().length > 3) : [];
  const ingredients = r.ingredients ? r.ingredients.split(/[,\s]+and\s+|\s*,\s*/).filter(Boolean) : [];
  const doctors = r.doctor_type ? r.doctor_type.split(/[\s,\/]+/).filter(Boolean) : [];

  useEffect(() => {
    const handleKey = e => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, []);

  return (
    <div onClick={onClose} style={{ position: "fixed", inset: 0, background: "rgba(44,24,16,0.6)", display: "flex", alignItems: "center", justifyContent: "center", zIndex: 1000, padding: "1.5rem" }}>
      <div className="slide-up" onClick={e => e.stopPropagation()}
        style={{ background: "#FDFAF5", borderRadius: 18, maxWidth: 700, width: "100%", maxHeight: "85vh", overflowY: "auto", position: "relative", boxShadow: "0 24px 64px rgba(44,24,16,0.35)" }}>

        {/* Modal Header */}
        <div style={{ background: `linear-gradient(135deg, ${cs.bg}, #FFF)`, padding: "1.5rem 1.5rem 1rem", borderBottom: "1px solid #E8D9C5", borderRadius: "18px 18px 0 0", position: "sticky", top: 0, zIndex: 10 }}>
          <button onClick={onClose} className="close-btn"
            style={{ position: "absolute", top: 14, right: 14, background: "rgba(0,0,0,0.05)", border: "none", borderRadius: 8, width: 32, height: 32, cursor: "pointer", fontSize: 18, color: "#8B5C2A", display: "flex", alignItems: "center", justifyContent: "center" }}>
            ×
          </button>

          <div style={{ display: "flex", gap: 14, alignItems: "flex-start", paddingRight: 40 }}>
            <div style={{ width: 52, height: 52, background: cs.bg, border: `2px solid ${cs.border}`, borderRadius: 14, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 24, flexShrink: 0 }}>
              {icon}
            </div>
            <div>
              <h2 style={{ margin: "0 0 6px", fontSize: 24, fontWeight: 500, color: "#2C1810", fontFamily: "'Cormorant Garamond', serif", lineHeight: 1.2 }}>
                {r.formulation_name}
              </h2>
              <div style={{ display: "flex", gap: 6, flexWrap: "wrap", alignItems: "center" }}>
                {r.category && (
                  <span style={{ fontSize: 11, fontFamily: "'Nunito', sans-serif", fontWeight: 700, textTransform: "uppercase", letterSpacing: 1, color: cs.text, background: cs.bg, padding: "3px 10px", borderRadius: 99, border: `1px solid ${cs.border}` }}>
                    {r.category}
                  </span>
                )}
                {doctors.map(d => <span key={d} className="tag-doctor">{d}</span>)}
              </div>
            </div>
          </div>
        </div>

        {/* Modal Body */}
        <div style={{ padding: "1.25rem 1.5rem", display: "flex", flexDirection: "column", gap: 20 }}>

          {/* Description */}
          {r.description && (
            <Section icon="📜" title="Description">
              <p style={{ margin: 0, fontSize: 15, color: "#4A2C14", lineHeight: 1.8, fontStyle: "italic" }}>{r.description}</p>
            </Section>
          )}

          {/* Symptoms */}
          {r.symptoms && (
            <Section icon="🩺" title="Indicated Symptoms">
              <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                {r.symptoms.split(" ").filter(Boolean).map((s, i) => (
                  <span key={i} style={{ fontSize: 12, fontFamily: "'Nunito', sans-serif", background: "#FFF3E8", color: "#7A3A0A", padding: "4px 12px", borderRadius: 99, border: "1px solid #F4C96A" }}>
                    {s}
                  </span>
                ))}
              </div>
            </Section>
          )}
          
          {/* Dosha Section */}
          {r.dosha && (
            <Section icon="🍃" title="Dosha">
              <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                {r.dosha.map((d, i) => (
                  <span
                    key={i}
                    style={{
                      fontSize: 12,
                      padding: "4px 12px",
                      borderRadius: 999,
                      fontFamily: "'Nunito', sans-serif",
                      fontWeight: 600,

                      background:
                        d === "Vata" ? "#E8F1FF" :
                        d === "Pitta" ? "#FFEAEA" :
                        d === "Kapha" ? "#E9F7EF" : "#F5EFE6",

                      color:
                        d === "Vata" ? "#3A6FD8" :
                        d === "Pitta" ? "#C0392B" :
                        d === "Kapha" ? "#1E8449" : "#8B5C2A",

                      border:
                        d === "Vata" ? "1px solid #B7D0FF" :
                        d === "Pitta" ? "1px solid #F5B7B1" :
                        d === "Kapha" ? "1px solid #ABEBC6" : "1px solid #E0D6C8",
                    }}
                  >
                    {d}
                  </span>
                ))}
              </div>
            </Section>
          )}
          {/* Ingredients */}
          {r.ingredients && (
            <Section icon="🌱" title="Ingredients">
              <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                {r.ingredients.split(/,\s*|\s+and\s+/).filter(Boolean).map((ing, i) => (
                  <span key={i} style={{ fontSize: 12.5, fontFamily: "'Nunito', sans-serif", background: "#E6F7F0", color: "#0F5C3A", padding: "4px 12px", borderRadius: 99, border: "1px solid #6ABFA0" }}>
                    {ing.trim()}
                  </span>
                ))}
              </div>
            </Section>
          )}

          {/* Steps & Dosage side by side */}
          <div style={{ display: "grid", gridTemplateColumns: r.dosage ? "1fr 1fr" : "1fr", gap: 16 }}>
            {r.steps && (
              <Section icon="⚗️" title="Preparation Steps">
                <ol className="steps-list">
                  {r.steps.split(/\.\s+|\bstep\s+\d+[.:]\s*/i).filter(s => s.trim().length > 5).map((s, i) => (
                    <li key={i} className="step-item" style={{ display: "flex", alignItems: "flex-start", fontSize: 12.5, fontFamily: "'Nunito', sans-serif", color: "#4A2C14", lineHeight: 1.6, marginBottom: 8 }}>
                      {s.trim().replace(/\.$/, "")}.
                    </li>
                  ))}
                </ol>
              </Section>
            )}

            {r.dosage && (
              <Section icon="💊" title="Dosage">
                <div style={{ background: "#FFF8E6", border: "1px solid #F4C96A", borderRadius: 10, padding: "0.8rem 1rem" }}>
                  <p style={{ margin: 0, fontSize: 14, fontFamily: "'Nunito', sans-serif", color: "#7A4A0A", lineHeight: 1.7 }}>{r.dosage}</p>
                </div>
              </Section>
            )}
          </div>

          {/* Product link */}
          {r.product_link && r.product_link.startsWith("http") && (
            <Section icon="🛒" title="Product Link">
              <a href={r.product_link} target="_blank" rel="noreferrer"
                style={{ display: "inline-flex", alignItems: "center", gap: 6, padding: "8px 18px", background: "#8B5C2A", color: "#FDF8F0", borderRadius: 10, textDecoration: "none", fontSize: 13, fontFamily: "'Nunito', sans-serif", fontWeight: 700 }}>
                🔗 View Product
              </a>
            </Section>
          )}

          {/* Doctor recommendation */}
          {doctors.length > 0 && (
            <Section icon="👨‍⚕️" title="Consult a Specialist">
              <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
                {doctors.map(d => (
                  <div key={d} style={{ display: "flex", alignItems: "center", gap: 6, background: "#E8F4F8", border: "1px solid #A8D8EA", borderRadius: 10, padding: "6px 14px" }}>
                    <span style={{ fontSize: 14 }}>🩺</span>
                    <span style={{ fontSize: 13, fontFamily: "'Nunito', sans-serif", fontWeight: 600, color: "#1A5F7A", textTransform: "capitalize" }}>{d}</span>
                  </div>
                ))}
              </div>
            </Section>
          )}
        </div>
      </div>
    </div>
  );
}

function Section({ icon, title, children }) {
  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", gap: 7, marginBottom: 10 }}>
        <span style={{ fontSize: 14 }}>{icon}</span>
        <h4 style={{ margin: 0, fontSize: 11, fontFamily: "'Nunito', sans-serif", fontWeight: 700, textTransform: "uppercase", letterSpacing: 1.2, color: "#8B5C2A" }}>{title}</h4>
      </div>
      {children}
    </div>
  );
}
