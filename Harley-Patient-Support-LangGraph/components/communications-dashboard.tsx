// components/communications-dashboard.tsx (UPDATED)
"use client";

import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { CheckCircle, Clock, MessageSquare, Calendar } from "lucide-react";
import { ProfileChecklistDisplay } from "@/components/profile-checklist-display";

interface Communication {
  _id: string;
  patientId: string;
  type: "clinical" | "faq" | "personal" | "emergency";
  summary: string;
  severity?: string;
  status: "read" | "unread" | "pending" | "completed";
  createdAt: string;
  timestamp?: string;
  messageCount?: number;
}

interface CommunicationsDashboardProps {
  patientId: string;
  patientEmail?: string;
}

export function CommunicationsDashboard({
  patientId,
  patientEmail,
}: CommunicationsDashboardProps) {
  const [communications, setCommunications] = useState<Communication[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<string>("all");
  const [typeFilter, setTypeFilter] = useState<string>("all");
  const [expandedSummaries, setExpandedSummaries] = useState<Set<string>>(
    new Set()
  );

  useEffect(() => {
    const fetchCommunications = async () => {
      try {
        console.log(
          "[Patient Communications] Fetching summaries from chat_history collection"
        );
        const response = await fetch(
          `/api/chat-history/summaries?patientId=${patientId}&userRole=patient&type=${
            typeFilter === "all" ? "all" : typeFilter
          }`
        );
        const data = await response.json();

        console.log("[Patient Communications] Fetched summaries:", {
          count: data.communications?.length || 0,
          source: "chat_history collection only",
        });

        setCommunications(data.communications || []);
      } catch (error) {
        console.error("Failed to fetch communications:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchCommunications();
  }, [patientId, typeFilter]);

  const filteredCommunications =
    filter === "all"
      ? communications
      : communications.filter((comm) => comm.severity === filter);

  const toggleSummaryExpansion = (id: string) => {
    setExpandedSummaries((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(id)) {
        newSet.delete(id);
      } else {
        newSet.add(id);
      }
      return newSet;
    });
  };

  const formatSummary = (text: string) => {
    if (!text) return null;

    const formatted = text.replace(/\*\*/g, "");
    const sections = formatted.split("\n\n");

    return sections.map((section, idx) => {
      const lines = section.split("\n");
      const firstLine = lines[0];

      const isHeader =
        firstLine.endsWith(":") ||
        (firstLine.length < 50 && firstLine === firstLine.toUpperCase());

      if (isHeader && lines.length > 1) {
        return (
          <div key={idx} className="mb-4">
            <h4 className="font-semibold text-gray-900 mb-2 text-sm">
              {firstLine}
            </h4>
            <div className="ml-3 space-y-1">
              {lines.slice(1).map((line, lineIdx) => {
                if (!line.trim()) return null;

                const isBullet = line.trim().match(/^(\d+\.|•|-|\*)/);
                const cleanLine = line.replace(/^(\s*(\d+\.|•|-|\*)\s*)/, "");

                return (
                  <div key={lineIdx} className={isBullet ? "flex gap-2" : ""}>
                    {isBullet && (
                      <span className="text-gray-600 text-sm">•</span>
                    )}
                    <p className="text-gray-700 text-sm">{cleanLine}</p>
                  </div>
                );
              })}
            </div>
          </div>
        );
      } else {
        return (
          <div key={idx} className="mb-3">
            {lines.map((line, lineIdx) => {
              if (!line.trim()) return null;

              const isBullet = line.trim().match(/^(\d+\.|•|-|\*)/);
              const cleanLine = line.replace(/^(\s*(\d+\.|•|-|\*)\s*)/, "");

              if (isBullet) {
                return (
                  <div key={lineIdx} className="flex gap-2 mb-1">
                    <span className="text-gray-600 text-sm">•</span>
                    <p className="text-gray-700 text-sm">{cleanLine}</p>
                  </div>
                );
              }

              return (
                <p key={lineIdx} className="text-gray-700 text-sm mb-1">
                  {line}
                </p>
              );
            })}
          </div>
        );
      }
    });
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "completed":
        return <CheckCircle className="w-4 h-4 text-green-600" />;
      case "in_progress":
        return <Clock className="w-4 h-4 text-yellow-600" />;
      default:
        return <MessageSquare className="w-4 h-4 text-blue-600" />;
    }
  };

  const getSeverityColor = (severity?: string) => {
    switch (severity) {
      case "critical":
        return "destructive";
      case "high":
        return "secondary";
      case "medium":
        return "default";
      case "low":
        return "outline";
      default:
        return "outline";
    }
  };

  const getTypeColor = (type: string) => {
    switch (type) {
      case "emergency":
        return "destructive";
      case "clinical":
        return "default";
      case "faq":
        return "outline";
      case "personal":
        return "secondary";
      default:
        return "outline";
    }
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <MessageSquare className="w-5 h-5" />
          Conversation Summaries with Profile Analysis
        </CardTitle>
        <p className="text-sm text-gray-600 mt-2">
          Your conversation summaries with AI-powered profile comparison and
          documentation quality analysis
        </p>
        <div className="flex gap-2 mt-4 flex-wrap">
          <div className="flex gap-2 items-center">
            <label className="text-sm font-medium text-gray-700">Type:</label>
            {["all", "clinical", "faq", "personal"].map((t) => (
              <button
                key={t}
                onClick={() => setTypeFilter(t)}
                className={`px-3 py-1 rounded text-sm font-medium transition ${
                  typeFilter === t
                    ? "bg-blue-600 text-white"
                    : "bg-gray-200 text-gray-700 hover:bg-gray-300"
                }`}
              >
                {t.charAt(0).toUpperCase() + t.slice(1)}
              </button>
            ))}
          </div>

          <div className="flex gap-2 items-center">
            <label className="text-sm font-medium text-gray-700">
              Severity:
            </label>
            {["all", "low", "medium", "high", "critical"].map((f) => (
              <button
                key={f}
                onClick={() => setFilter(f)}
                className={`px-3 py-1 rounded text-sm font-medium transition ${
                  filter === f
                    ? "bg-blue-600 text-white"
                    : "bg-gray-200 text-gray-700 hover:bg-gray-300"
                }`}
              >
                {f.charAt(0).toUpperCase() + f.slice(1)}
              </button>
            ))}
          </div>
        </div>
      </CardHeader>

      <CardContent>
        {loading ? (
          <div className="flex items-center justify-center py-8">
            <div className="text-center">
              <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-blue-600 mx-auto"></div>
              <p className="mt-3 text-gray-500">Loading summaries...</p>
            </div>
          </div>
        ) : filteredCommunications.length === 0 ? (
          <div className="text-center py-8">
            <MessageSquare className="w-12 h-12 text-gray-300 mx-auto mb-3" />
            <p className="text-gray-500">
              No conversation summaries yet. Complete a chat and create a
              summary to see it here.
            </p>
          </div>
        ) : (
          <div className="space-y-4">
            {filteredCommunications.map((comm) => (
              <div
                key={comm._id}
                className="border border-gray-200 rounded-lg p-5 hover:shadow-md transition bg-white"
              >
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center gap-2 flex-wrap">
                    {getStatusIcon(comm.status)}
                    <Badge
                      variant={getTypeColor(comm.type)}
                      className="font-semibold"
                    >
                      {comm.type.toUpperCase()}
                    </Badge>
                    <Badge
                      variant={getSeverityColor(comm.severity)}
                      className="font-semibold"
                    >
                      {(comm.severity || "medium").toUpperCase()}
                    </Badge>
                  </div>
                  <div className="flex items-center gap-2 text-xs text-gray-500">
                    <Calendar className="w-3 h-3" />
                    {new Date(
                      comm.timestamp || comm.createdAt
                    ).toLocaleString()}
                  </div>
                </div>

                {comm.messageCount && (
                  <div className="mb-3 pb-3 border-b border-gray-100">
                    <p className="text-xs text-gray-500">
                      Conversation:{" "}
                      <span className="font-medium text-gray-700">
                        {comm.messageCount} messages
                      </span>
                    </p>
                  </div>
                )}

                <div className="bg-gray-50 p-4 rounded-lg border border-gray-200">
                  <div className="mb-3 pb-2 border-b border-gray-300">
                    <h4 className="text-sm font-semibold text-gray-900">
                      Medical Summary
                    </h4>
                  </div>
                  <div className="prose prose-sm max-w-none">
                    {formatSummary(comm.summary)}
                  </div>
                </div>

                {/* AI-Powered Profile Checklist Section */}
                <div className="mt-4 pt-4 border-t border-gray-200">
                  <button
                    onClick={() => toggleSummaryExpansion(comm._id)}
                    className="text-sm font-medium text-blue-600 hover:text-blue-700 flex items-center gap-2 mb-3"
                  >
                    {expandedSummaries.has(comm._id) ? "▼" : "▶"} View AI
                    Profile Comparison & Documentation Analysis
                  </button>

                  {expandedSummaries.has(comm._id) && (
                    <div className="mt-3">
                      <ProfileChecklistDisplay
                        summary={comm.summary}
                        patientId={comm.patientId}
                        patientEmail={patientEmail}
                        showStats={true}
                      />
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
