// components/profile-checklist-display.tsx
"use client";

import { useState, useEffect } from "react";
import {
  CheckCircle2,
  XCircle,
  AlertCircle,
  RefreshCw,
  Loader2,
  Info,
} from "lucide-react";
import {
  generateProfileChecklist,
  groupProfileChecklistByCategory,
  getProfileComparisonStats,
  getStatusColor,
  getStatusIcon,
  type ProfileChecklistItem,
} from "@/lib/profile-checklist-generator";

interface ProfileChecklistDisplayProps {
  summary: string;
  patientId: string;
  patientEmail?: string;
  showStats?: boolean;
}

export function ProfileChecklistDisplay({
  summary,
  patientId,
  patientEmail,
  showStats = true,
}: ProfileChecklistDisplayProps) {
  const [checklist, setChecklist] = useState<ProfileChecklistItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;

    const loadChecklist = async () => {
      if (!summary || summary.trim().length === 0 || !patientId) {
        setLoading(false);
        return;
      }

      setLoading(true);
      setError(null);

      try {
        console.log(
          "[Profile Checklist Display] Generating checklist with profile data..."
        );
        const items = await generateProfileChecklist(
          summary,
          patientId,
          patientEmail
        );

        if (mounted) {
          setChecklist(items);
          console.log(
            "[Profile Checklist Display] Checklist loaded:",
            items.length,
            "items"
          );
        }
      } catch (err) {
        console.error(
          "[Profile Checklist Display] Error loading checklist:",
          err
        );
        if (mounted) {
          setError("Failed to generate profile checklist.");
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
  }, [summary, patientId, patientEmail]);

  if (loading) {
    return (
      <div className="bg-gray-50 border border-gray-200 rounded-lg p-6 text-center">
        <Loader2 className="w-8 h-8 animate-spin text-blue-600 mx-auto mb-3" />
        <p className="text-sm text-gray-600">
          AI is comparing summary with patient profile...
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
              Profile Comparison Issue
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
          No profile checklist items generated
        </p>
      </div>
    );
  }

  const groupedChecklist = groupProfileChecklistByCategory(checklist);
  const stats = showStats ? getProfileComparisonStats(checklist) : null;

  return (
    <div className="space-y-4">
      {/* Header with AI Badge */}
      <div className="bg-gradient-to-r from-purple-50 via-blue-50 to-green-50 border border-purple-200 rounded-lg p-4">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <h4 className="text-sm font-semibold text-gray-900">
              AI Profile Comparison Analysis
            </h4>
            <span className="text-xs font-medium text-purple-600 bg-purple-100 px-2 py-0.5 rounded-full">
              ✨ Profile-Based AI
            </span>
          </div>
        </div>
        <p className="text-xs text-gray-600">
          Comparing conversation summary with existing patient profile data from
          questionnaires and medical history
        </p>
        {stats && (
          <div className="flex items-center gap-4 text-xs text-gray-600 mt-3 pt-3 border-t border-purple-200">
            <span className="flex items-center gap-1">
              <CheckCircle2 className="w-3 h-3 text-green-600" />
              {stats.present} Present
            </span>
            <span className="flex items-center gap-1">
              <RefreshCw className="w-3 h-3 text-blue-600" />
              {stats.updated} Updated
            </span>
            <span className="flex items-center gap-1">
              <XCircle className="w-3 h-3 text-red-600" />
              {stats.missing} Missing
            </span>
            <span className="flex items-center gap-1">
              <AlertCircle className="w-3 h-3 text-yellow-600" />
              {stats.partial} Partial
            </span>
            <span className="ml-auto font-semibold text-gray-900">
              {stats.completeness}% Complete
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
                {items.filter((i) => i.checked).length}/{items.length} verified
              </span>
            </h5>
            <div className="space-y-3">
              {items.map((item) => (
                <div
                  key={item.id}
                  className="group hover:bg-gray-50 p-3 rounded transition-colors border border-gray-100"
                >
                  <div className="flex items-start gap-3">
                    {/* Status Icon */}
                    <div className="flex-shrink-0 mt-0.5">
                      {item.status === "present" && (
                        <CheckCircle2 className="w-5 h-5 text-green-600" />
                      )}
                      {item.status === "updated" && (
                        <RefreshCw className="w-5 h-5 text-blue-600" />
                      )}
                      {item.status === "missing" && (
                        <XCircle className="w-5 h-5 text-red-600" />
                      )}
                      {item.status === "partial" && (
                        <AlertCircle className="w-5 h-5 text-yellow-600" />
                      )}
                    </div>

                    {/* Content */}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-start justify-between gap-2 mb-2">
                        <span className="text-sm font-medium text-gray-900">
                          {item.label}
                        </span>
                        <span
                          className={`inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-full border flex-shrink-0 ${getStatusColor(
                            item.status
                          )}`}
                        >
                          {getStatusIcon(item.status)}{" "}
                          {item.status.toUpperCase()}
                        </span>
                      </div>

                      {/* Importance badge */}
                      <div className="flex items-center gap-2 mb-2">
                        <span
                          className={`inline-block text-xs px-2 py-0.5 rounded-full border ${
                            item.importance === "critical"
                              ? "text-red-600 bg-red-50 border-red-200"
                              : item.importance === "high"
                              ? "text-orange-600 bg-orange-50 border-orange-200"
                              : item.importance === "medium"
                              ? "text-blue-600 bg-blue-50 border-blue-200"
                              : "text-gray-600 bg-gray-50 border-gray-200"
                          }`}
                        >
                          {item.importance.charAt(0).toUpperCase() +
                            item.importance.slice(1)}{" "}
                          Priority
                        </span>
                      </div>

                      {/* Profile vs Summary values */}
                      {(item.profileValue || item.summaryValue) && (
                        <div className="bg-gray-50 rounded p-2 space-y-1 text-xs">
                          {item.profileValue && (
                            <div className="flex items-start gap-2">
                              <span className="text-gray-600 font-medium min-w-[80px]">
                                Profile:
                              </span>
                              <span className="text-gray-800">
                                {item.profileValue}
                              </span>
                            </div>
                          )}
                          {item.summaryValue && (
                            <div className="flex items-start gap-2">
                              <span className="text-gray-600 font-medium min-w-[80px]">
                                Summary:
                              </span>
                              <span className="text-gray-800">
                                {item.summaryValue}
                              </span>
                            </div>
                          )}
                        </div>
                      )}

                      {/* Notes */}
                      {item.notes && (
                        <div className="flex items-start gap-2 mt-2 text-xs text-gray-600 bg-blue-50 rounded p-2">
                          <Info className="w-3 h-3 flex-shrink-0 mt-0.5" />
                          <span>{item.notes}</span>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Overall Progress Bar */}
      {/* {stats && (
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-medium text-gray-600">
              Profile Data Completeness
            </span>
            <span className="text-xs font-semibold text-gray-900">
              {stats.completeness}%
            </span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2.5">
            <div
              className={`h-2.5 rounded-full transition-all duration-500 ${
                stats.completeness === 100
                  ? "bg-green-600"
                  : stats.completeness >= 80
                  ? "bg-blue-600"
                  : stats.completeness >= 60
                  ? "bg-yellow-600"
                  : "bg-red-600"
              }`}
              style={{ width: `${stats.completeness}%` }}
            ></div>
          </div>
          <p className="text-xs text-gray-500 mt-2 text-center">
            AI-powered comparison of profile data with conversation summary
          </p>
        </div>
      )} */}
    </div>
  );
}
