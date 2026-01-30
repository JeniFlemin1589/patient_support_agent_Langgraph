"use client";

import { useState, useEffect } from "react";
import { CheckCircle2, Circle, Loader2, AlertCircle } from "lucide-react";
import {
  generateAIChecklist,
  groupChecklistByCategory,
  getChecklistCompletionPercentage,
  getSummaryStatistics,
  getImportanceColor,
  type AIChecklistItem,
} from "@/lib/checklist-generator";

interface AIChecklistDisplayProps {
  summary: string;
  showStats?: boolean;
}

export function AIChecklistDisplay({
  summary,
  showStats = true,
}: AIChecklistDisplayProps) {
  const [checklist, setChecklist] = useState<AIChecklistItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;

    const loadChecklist = async () => {
      if (!summary || summary.trim().length === 0) {
        setLoading(false);
        return;
      }

      setLoading(true);
      setError(null);

      try {
        console.log("[AIChecklistDisplay] Generating AI checklist...");
        const items = await generateAIChecklist(summary);

        if (mounted) {
          setChecklist(items);
          console.log(
            "[AIChecklistDisplay] Checklist loaded:",
            items.length,
            "items"
          );
        }
      } catch (err) {
        console.error("[AIChecklistDisplay] Error loading checklist:", err);
        if (mounted) {
          setError("Failed to generate AI checklist. Using fallback.");
        }
      } finally {
        if (mounted) {
          setLoading(false);
        }
      }
    };

    loadChecklist();

    return () => {
      mounted = false;
    };
  }, [summary]);

  if (loading) {
    return (
      <div className="bg-gray-50 border border-gray-200 rounded-lg p-6 text-center">
        <Loader2 className="w-8 h-8 animate-spin text-blue-600 mx-auto mb-3" />
        <p className="text-sm text-gray-600">
          AI is analyzing documentation quality...
        </p>
      </div>
    );
  }

  if (error && checklist.length === 0) {
    return (
      <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
        <div className="flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-yellow-600 flex-shrink-0 mt-0.5" />
          <div>
            <p className="text-sm font-medium text-yellow-900">
              Checklist Generation Issue
            </p>
            <p className="text-sm text-yellow-700 mt-1">{error}</p>
          </div>
        </div>
      </div>
    );
  }

  if (checklist.length === 0) {
    return (
      <div className="bg-gray-50 border border-gray-200 rounded-lg p-4 text-center">
        <p className="text-sm text-gray-500">
          No checklist items generated for this summary
        </p>
      </div>
    );
  }

  const groupedChecklist = groupChecklistByCategory(checklist);
  const completionPercentage = getChecklistCompletionPercentage(checklist);
  const stats = showStats ? getSummaryStatistics(summary) : null;

  return (
    <div className="space-y-4">
      {/* Header with AI Badge */}
      <div className="bg-gradient-to-r from-blue-50 to-purple-50 border border-blue-200 rounded-lg p-4">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <h4 className="text-sm font-semibold text-gray-900">
              AI Documentation Quality Analysis
            </h4>
            <span className="text-xs font-medium text-purple-600 bg-purple-100 px-2 py-0.5 rounded-full">
              ✨ AI Generated
            </span>
          </div>
          <span className="text-xs font-medium text-blue-600 bg-blue-100 px-2 py-1 rounded-full">
            {completionPercentage}% Complete
          </span>
        </div>
        {stats && (
          <div className="flex items-center gap-4 text-xs text-gray-600 mt-2">
            <span>📝 {stats.wordCount} words</span>
            <span>⏱️ {stats.estimatedReadingTime} min read</span>
            <span>
              {stats.hasStructuredFormat ? "✅" : "📋"} Structured format
            </span>
          </div>
        )}
      </div>

      {/* Checklist Items by Category */}
      <div className="space-y-4">
        {Object.entries(groupedChecklist).map(([category, items]) => (
          <div
            key={category}
            className="bg-white border border-gray-200 rounded-lg p-4"
          >
            <h5 className="text-sm font-semibold text-gray-800 mb-3 pb-2 border-b border-gray-200 flex items-center justify-between">
              <span>{category}</span>
              <span className="text-xs text-gray-500 font-normal">
                {items.filter((i) => i.checked).length}/{items.length} complete
              </span>
            </h5>
            <div className="space-y-2">
              {items.map((item) => (
                <div
                  key={item.id}
                  className="flex items-start gap-3 group hover:bg-gray-50 p-2 rounded transition-colors"
                >
                  <div className="flex-shrink-0 mt-0.5">
                    {item.checked ? (
                      <CheckCircle2 className="w-5 h-5 text-green-600" />
                    ) : (
                      <Circle className="w-5 h-5 text-gray-300" />
                    )}
                  </div>
                  <div className="flex-1 min-w-0">
                    <span
                      className={`text-sm block ${
                        item.checked
                          ? "text-gray-700"
                          : "text-gray-400 line-through"
                      }`}
                    >
                      {item.label}
                    </span>
                    {/* Importance badge */}
                    <span
                      className={`inline-block text-xs px-2 py-0.5 rounded-full mt-1 border ${getImportanceColor(
                        item.importance
                      )}`}
                    >
                      {item.importance.charAt(0).toUpperCase() +
                        item.importance.slice(1)}{" "}
                      Priority
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Progress Bar */}
      <div className="bg-white border border-gray-200 rounded-lg p-4">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-medium text-gray-600">
            Overall Documentation Quality
          </span>
          <span className="text-xs font-semibold text-gray-900">
            {checklist.filter((item) => item.checked).length} of{" "}
            {checklist.length} items documented
          </span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2.5">
          <div
            className={`h-2.5 rounded-full transition-all duration-500 ${
              completionPercentage === 100
                ? "bg-green-600"
                : completionPercentage >= 75
                ? "bg-blue-600"
                : completionPercentage >= 50
                ? "bg-yellow-600"
                : "bg-orange-600"
            }`}
            style={{ width: `${completionPercentage}%` }}
          ></div>
        </div>
        <p className="text-xs text-gray-500 mt-2 text-center">
          AI-powered analysis of documentation completeness and quality
        </p>
      </div>
    </div>
  );
}
