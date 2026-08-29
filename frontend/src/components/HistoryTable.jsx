import React, { useState, useEffect } from "react";
import { History, Search, Trash2, Eye, RefreshCw, ChevronLeft, ChevronRight, AlertCircle, User, Globe } from "lucide-react";
import { getResults, deleteResult, getAssetUrl, formatLocalTimestamp } from "../api/client";
import DetailModal from "./DetailModal";

const FILTER_TABS = ["ALL", "ACCEPTABLE", "DEGRADED", "DEFECTIVE"];

export default function HistoryTable() {
  const [items, setItems] = useState([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [limit] = useState(10);
  const [totalPages, setTotalPages] = useState(1);
  const [filterLabel, setFilterLabel] = useState("ALL");
  const [scope, setScope] = useState("session"); // "session" or "global"
  const [searchQuery, setSearchQuery] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [selectedItem, setSelectedItem] = useState(null);
  const [deletingId, setDeletingId] = useState(null);

  const fetchHistory = async () => {
    setIsLoading(true);
    try {
      const data = await getResults(page, limit, filterLabel, scope);
      setItems(data.items || []);
      setTotal(data.total || 0);
      setTotalPages(data.pages || 1);
    } catch (err) {
      console.error("Failed to fetch history:", err);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, [page, filterLabel, scope]);

  const handleDelete = async (id, e) => {
    e.stopPropagation();
    if (!window.confirm(`Delete analysis record #${id}?`)) return;
    try {
      setDeletingId(id);
      await deleteResult(id);
      setItems((prev) => prev.filter((it) => it.id !== id));
      setTotal((prev) => Math.max(0, prev - 1));
    } catch (err) {
      alert("Failed to delete record.");
    } finally {
      setDeletingId(null);
    }
  };

  const filteredItems = items.filter((it) =>
    it.filename.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="history-section">
      <div className="workbench-panel">
        {/* Table Controls Bar */}
        <div className="history-controls-bar" style={{ display: "flex", flexWrap: "wrap", gap: "0.75rem", justifyContent: "space-between", alignItems: "center" }}>
          {/* Scope Selector: My Session vs Global Feed */}
          <div className="filter-tabs-group mono" style={{ display: "flex", gap: "0.25rem" }}>
            <button
              className={`filter-tab ${scope === "session" ? "active" : ""}`}
              onClick={() => {
                setScope("session");
                setPage(1);
              }}
              title="Show records analyzed in your current browser session"
              style={{ display: "inline-flex", alignItems: "center", gap: "0.35rem" }}
            >
              <User size={13} />
              <span>My Session</span>
            </button>
            <button
              className={`filter-tab ${scope === "global" ? "active" : ""}`}
              onClick={() => {
                setScope("global");
                setPage(1);
              }}
              title="Show all benchmark & historical records across all sessions"
              style={{ display: "inline-flex", alignItems: "center", gap: "0.35rem" }}
            >
              <Globe size={13} />
              <span>Global Feed</span>
            </button>
          </div>

          {/* Status Filter Tabs (Square tabs) */}
          <div className="filter-tabs-group mono">
            {FILTER_TABS.map((tab) => (
              <button
                key={tab}
                className={`filter-tab ${filterLabel === tab ? "active" : ""}`}
                onClick={() => {
                  setFilterLabel(tab);
                  setPage(1);
                }}
              >
                {tab}
              </button>
            ))}
          </div>

          {/* Search Bar & Refresh */}
          <div className="history-search-group">
            <div className="search-input-box mono">
              <Search size={14} className="text-secondary" />
              <input
                type="text"
                placeholder="Filter by filename..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="search-input"
              />
            </div>
            <button className="btn btn-secondary" onClick={fetchHistory} disabled={isLoading}>
              <RefreshCw size={14} className={isLoading ? "spin" : ""} />
              <span>Refresh</span>
            </button>
          </div>
        </div>

        {/* Data Table */}
        <div className="table-responsive">
          <table className="audit-table mono">
            <thead>
              <tr>
                <th style={{ width: "70px" }}>Preview</th>
                <th>Record ID & Filename</th>
                <th>Score</th>
                <th>Classification</th>
                <th>Detected Issues</th>
                <th>Timestamp</th>
                <th style={{ textAlign: "right" }}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {isLoading && items.length === 0 ? (
                <tr>
                  <td colSpan={7} className="table-empty-cell">
                    <RefreshCw size={20} className="spin text-highlight" style={{ margin: "0 auto 0.5rem" }} />
                    <div>Loading historical records...</div>
                  </td>
                </tr>
              ) : filteredItems.length === 0 ? (
                <tr>
                  <td colSpan={7} className="table-empty-cell">
                    <AlertCircle size={20} className="text-muted" style={{ margin: "0 auto 0.5rem" }} />
                    <div>No analysis records found matching current criteria.</div>
                  </td>
                </tr>
              ) : (
                filteredItems.map((row) => (
                  <tr key={row.id} onClick={() => setSelectedItem(row)} className="clickable-row">
                    <td>
                      <div className="table-thumb-box">
                        <img
                          src={getAssetUrl(row.image_url)}
                          alt={row.filename}
                          className="table-thumb"
                          loading="lazy"
                        />
                      </div>
                    </td>
                    <td>
                      <div className="filename-cell">
                        <span className="file-name">{row.filename}</span>
                        <span className="file-id text-muted">ID: #{row.id}</span>
                      </div>
                    </td>
                    <td>
                      <span className="table-score-val text-highlight">
                        {row.quality_score?.toFixed(1) ?? "—"}
                      </span>
                    </td>
                    <td>
                      <span className={`status-tag ${row.quality_label?.toLowerCase()}`}>
                        {row.quality_label}
                      </span>
                    </td>
                    <td>
                      <div className="table-issues-chips">
                        {row.issues && row.issues.length > 0 ? (
                          row.issues.map((iss, i) => (
                            <span key={i} className="issue-micro-chip">
                              {iss.type} ({Math.round(iss.confidence * 100)}%)
                            </span>
                          ))
                        ) : (
                          <span className="text-muted">None (Clean)</span>
                        )}
                      </div>
                    </td>
                    <td>
                      <span className="text-secondary text-sm">
                        {row.processed_at ? formatLocalTimestamp(row.processed_at, "full") : "—"}
                      </span>
                    </td>
                    <td style={{ textAlign: "right" }}>
                      <div className="table-action-btns">
                        <button
                          className="btn btn-secondary btn-sm"
                          onClick={(e) => {
                            e.stopPropagation();
                            setSelectedItem(row);
                          }}
                          title="Inspect Detailed Diagnostics"
                        >
                          <Eye size={13} />
                          <span>Inspect</span>
                        </button>
                        <button
                          className="btn btn-danger btn-sm"
                          onClick={(e) => handleDelete(row.id, e)}
                          disabled={deletingId === row.id}
                          title="Delete Record"
                        >
                          <Trash2 size={13} />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination Footer */}
        <div className="table-pagination-footer mono">
          <span className="pagination-count text-secondary">
            Showing {filteredItems.length} of {total} total records
          </span>
          <div className="pagination-btns">
            <button
              className="btn btn-secondary btn-sm"
              disabled={page <= 1 || isLoading}
              onClick={() => setPage((p) => Math.max(1, p - 1))}
            >
              <ChevronLeft size={14} />
              <span>Previous</span>
            </button>
            <span className="page-indicator">
              Page {page} of {totalPages}
            </span>
            <button
              className="btn btn-secondary btn-sm"
              disabled={page >= totalPages || isLoading}
              onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
            >
              <span>Next</span>
              <ChevronRight size={14} />
            </button>
          </div>
        </div>
      </div>

      {/* Full Detail Modal */}
      {selectedItem && (
        <DetailModal item={selectedItem} onClose={() => setSelectedItem(null)} />
      )}
    </div>
  );
}
