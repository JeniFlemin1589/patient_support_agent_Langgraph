// components/doctor-dashboard.tsx (UPDATED)
"use client";

import { useState, useEffect } from "react";
import {
  Search,
  Filter,
  Mail,
  Calendar,
  AlertCircle,
  User,
  Phone,
} from "lucide-react";
import { ProfileChecklistDisplay } from "@/components/profile-checklist-display";

interface Communication {
  _id: string;
  patientId: string;
  patientEmail?: string;
  type: string;
  question: string;
  answer: string;
  summary?: string;
  severity?: string;
  status: string;
  createdAt: string;
  timestamp?: string;
  messageCount?: number;
}

interface PatientInfo {
  email: string;
  name: string;
  contact: string;
  createdAt: string;
}

interface DoctorDashboardProps {
  doctorEmail: string;
}

export function DoctorDashboard({ doctorEmail }: DoctorDashboardProps) {
  const [communications, setCommunications] = useState<Communication[]>([]);
  const [filteredComms, setFilteredComms] = useState<Communication[]>([]);
  const [patientInfo, setPatientInfo] = useState<Map<string, PatientInfo>>(
    new Map()
  );
  const [loading, setLoading] = useState(true);
  const [emailFilter, setEmailFilter] = useState("");
  const [typeFilter, setTypeFilter] = useState("all");
  const [severityFilter, setSeverityFilter] = useState("all");
  const [expandedSummaries, setExpandedSummaries] = useState<Set<string>>(
    new Set()
  );

  useEffect(() => {
    fetchCommunications();
  }, []);

  useEffect(() => {
    applyFilters();
  }, [communications, emailFilter, typeFilter, severityFilter]);

  const fetchCommunications = async () => {
    try {
      setLoading(true);
      const response = await fetch("/api/doctor/communications");
      const data = await response.json();

      console.log(
        "[Doctor Dashboard] Fetched summaries from chat_history collection:",
        {
          count: data.communications?.length || 0,
          source: "chat_history collection only",
        }
      );

      if (data.success) {
        setCommunications(data.communications);

        // Fetch patient info for each communication
        data.communications.forEach(async (comm: Communication) => {
          if (comm.patientEmail && !patientInfo.has(comm.patientId)) {
            try {
              const patResponse = await fetch(
                `/api/patient/profile?email=${encodeURIComponent(
                  comm.patientEmail
                )}`
              );
              if (patResponse.ok) {
                const patData = await patResponse.json();
                setPatientInfo((prev) =>
                  new Map(prev).set(comm.patientId, patData.data)
                );
              }
            } catch (error) {
              console.error("Failed to fetch patient info:", error);
            }
          }
        });
      }
    } catch (error) {
      console.error("Failed to fetch communications:", error);
    } finally {
      setLoading(false);
    }
  };

  const applyFilters = () => {
    let filtered = [...communications];

    if (emailFilter.trim()) {
      filtered = filtered.filter(
        (comm) =>
          comm.patientId
            .toLowerCase()
            .includes(emailFilter.toLowerCase().replace(/[^a-zA-Z0-9]/g, "")) ||
          comm.patientEmail?.toLowerCase().includes(emailFilter.toLowerCase())
      );
    }

    if (typeFilter && typeFilter !== "all") {
      filtered = filtered.filter((comm) => comm.type === typeFilter);
    }

    if (severityFilter !== "all") {
      filtered = filtered.filter((comm) => comm.severity === severityFilter);
    }

    setFilteredComms(filtered);
  };

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
            <h4 className="font-semibold text-gray-900 mb-2">{firstLine}</h4>
            <div className="ml-4 space-y-1">
              {lines.slice(1).map((line, lineIdx) => {
                if (!line.trim()) return null;

                const isBullet = line.trim().match(/^(\d+\.|•|-|\*)/);
                const cleanLine = line.replace(/^(\s*(\d+\.|•|-|\*)\s*)/, "");

                return (
                  <div key={lineIdx} className={isBullet ? "flex gap-2" : ""}>
                    {isBullet && <span className="text-gray-600">•</span>}
                    <p className="text-gray-700">{cleanLine}</p>
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
                    <span className="text-gray-600">•</span>
                    <p className="text-gray-700">{cleanLine}</p>
                  </div>
                );
              }

              return (
                <p key={lineIdx} className="text-gray-700 mb-1">
                  {line}
                </p>
              );
            })}
          </div>
        );
      }
    });
  };

  const getSeverityColor = (severity?: string) => {
    switch (severity) {
      case "critical":
        return "bg-red-100 text-red-800 border-red-300";
      case "high":
        return "bg-orange-100 text-orange-800 border-orange-300";
      case "medium":
        return "bg-yellow-100 text-yellow-800 border-yellow-300";
      case "low":
        return "bg-green-100 text-green-800 border-green-300";
      default:
        return "bg-gray-100 text-gray-800 border-gray-300";
    }
  };

  const getTypeColor = (type: string) => {
    switch (type) {
      case "emergency":
        return "bg-red-600 text-white";
      case "clinical":
        return "bg-blue-600 text-white";
      case "personal":
        return "bg-purple-600 text-white";
      case "faq":
        return "bg-green-600 text-white";
      default:
        return "bg-gray-600 text-white";
    }
  };

  const uniquePatients = [...new Set(communications.map((c) => c.patientId))];

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading summaries...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6 p-6 bg-gray-50 min-h-screen">
      {/* Header Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Total Summaries</p>
              <p className="text-2xl font-bold text-gray-900">
                {communications.length}
              </p>
            </div>
            <Mail className="h-8 w-8 text-blue-600" />
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Total Patients</p>
              <p className="text-2xl font-bold text-gray-900">
                {uniquePatients.length}
              </p>
            </div>
            <User className="h-8 w-8 text-green-600" />
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">High Severity</p>
              <p className="text-2xl font-bold text-red-600">
                {
                  communications.filter(
                    (c) => c.severity === "high" || c.severity === "critical"
                  ).length
                }
              </p>
            </div>
            <AlertCircle className="h-8 w-8 text-red-600" />
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Filtered Results</p>
              <p className="text-2xl font-bold text-gray-900">
                {filteredComms.length}
              </p>
            </div>
            <Filter className="h-8 w-8 text-purple-600" />
          </div>
        </div>
      </div>

      {/* Filters */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
          <Filter className="h-5 w-5" />
          Filters
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Patient Email or ID
            </label>
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
              <input
                type="text"
                value={emailFilter}
                onChange={(e) => setEmailFilter(e.target.value)}
                placeholder="Search by email or ID..."
                className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-600 focus:border-transparent outline-none"
              />
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Communication Type
            </label>
            <select
              value={typeFilter}
              onChange={(e) => setTypeFilter(e.target.value)}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-600 focus:border-transparent outline-none"
            >
              <option value="all">All Types</option>
              <option value="clinical">Clinical</option>
              <option value="emergency">Emergency</option>
              <option value="faq">FAQ</option>
              <option value="personal">Personal</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Severity
            </label>
            <select
              value={severityFilter}
              onChange={(e) => setSeverityFilter(e.target.value)}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-600 focus:border-transparent outline-none"
            >
              <option value="all">All Severities</option>
              <option value="critical">Critical</option>
              <option value="high">High</option>
              <option value="medium">Medium</option>
              <option value="low">Low</option>
            </select>
          </div>
        </div>

        {(emailFilter || typeFilter !== "all" || severityFilter !== "all") && (
          <button
            onClick={() => {
              setEmailFilter("");
              setTypeFilter("all");
              setSeverityFilter("all");
            }}
            className="mt-4 text-sm text-blue-600 hover:text-blue-700 font-medium"
          >
            Clear all filters
          </button>
        )}
      </div>

      {/* Summaries List */}
      <div className="bg-white rounded-lg shadow">
        <div className="p-6 border-b border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900">
            Patient Summaries with Profile Analysis ({filteredComms.length})
          </h3>
        </div>

        <div className="divide-y divide-gray-200 max-h-[800px] overflow-y-auto">
          {filteredComms.length === 0 ? (
            <div className="p-8 text-center text-gray-500">
              No summaries found matching filters.
            </div>
          ) : (
            filteredComms.map((comm) => {
              const patient = patientInfo.get(comm.patientId);
              return (
                <div key={comm._id} className="p-6 hover:bg-gray-50 transition">
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex items-center gap-3">
                      <span
                        className={`px-3 py-1 rounded-full text-xs font-semibold ${getTypeColor(
                          comm.type
                        )}`}
                      >
                        {comm.type === "emergency" ? "🚨 " : ""}
                        {comm.type.toUpperCase().replace(/_/g, " ")}
                      </span>
                      {comm.severity && (
                        <span
                          className={`px-3 py-1 rounded-full text-xs font-semibold border ${getSeverityColor(
                            comm.severity
                          )}`}
                        >
                          {comm.severity.toUpperCase()}
                        </span>
                      )}
                    </div>
                    <div className="flex items-center gap-2 text-sm text-gray-500">
                      <Calendar className="h-4 w-4" />
                      {new Date(
                        comm.timestamp || comm.createdAt
                      ).toLocaleString()}
                    </div>
                  </div>

                  {/* Patient Information */}
                  {patient ? (
                    <div className="mb-4 bg-blue-50 rounded-lg p-4 border border-blue-200">
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                        <div>
                          <p className="text-xs text-gray-600 mb-1">
                            Patient Name
                          </p>
                          <p className="font-semibold text-gray-900">
                            {patient.name}
                          </p>
                        </div>
                        <div className="flex items-center gap-2">
                          <Mail className="h-4 w-4 text-gray-600 flex-shrink-0" />
                          <div className="min-w-0">
                            <p className="text-xs text-gray-600">Email</p>
                            <p className="text-sm text-gray-900 truncate">
                              {patient.email}
                            </p>
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          <Phone className="h-4 w-4 text-gray-600 flex-shrink-0" />
                          <div>
                            <p className="text-xs text-gray-600">Contact</p>
                            <p className="text-sm text-gray-900">
                              {patient.contact}
                            </p>
                          </div>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="mb-4 bg-gray-50 rounded-lg p-3 border border-gray-200">
                      <p className="text-sm text-gray-600">
                        Patient ID:{" "}
                        <span className="font-medium text-gray-900">
                          {comm.patientId}
                        </span>
                      </p>
                      <p className="text-sm text-gray-600">
                        Email:{" "}
                        <span className="font-medium text-gray-900">
                          {comm.patientEmail}
                        </span>
                      </p>
                    </div>
                  )}

                  {/* Medical Summary */}
                  <div className="bg-white rounded-lg border border-gray-200 p-5">
                    <div className="flex items-center justify-between mb-3 pb-2 border-b border-gray-200">
                      <h4 className="text-base font-semibold text-gray-900">
                        Medical Summary
                      </h4>
                      <span className="text-xs text-gray-500">
                        {comm.messageCount || "Multiple"} messages
                      </span>
                    </div>
                    <div className="prose prose-sm max-w-none">
                      {formatSummary(comm.summary || comm.answer)}
                    </div>
                  </div>

                  {/* AI Profile Comparison Section */}
                  <div className="mt-4 pt-4 border-t border-gray-200">
                    <button
                      onClick={() => toggleSummaryExpansion(comm._id)}
                      className="text-sm font-medium text-blue-600 hover:text-blue-700 flex items-center gap-2 mb-3"
                    >
                      {expandedSummaries.has(comm._id) ? "▼" : "▶"} View AI
                      Profile Comparison & Clinical Analysis
                    </button>

                    {expandedSummaries.has(comm._id) && (
                      <div className="mt-3">
                        <ProfileChecklistDisplay
                          summary={comm.summary || comm.answer}
                          patientId={comm.patientId}
                          patientEmail={comm.patientEmail}
                          showStats={true}
                        />
                      </div>
                    )}
                  </div>

                  <div className="mt-3 flex items-center gap-4 text-xs">
                    <span
                      className={`px-3 py-1 rounded-full font-medium ${
                        comm.status === "completed"
                          ? "bg-green-100 text-green-800"
                          : comm.status === "in_progress"
                          ? "bg-yellow-100 text-yellow-800"
                          : "bg-gray-100 text-gray-800"
                      }`}
                    >
                      Status: {comm.status.replace(/_/g, " ").toUpperCase()}
                    </span>
                  </div>
                </div>
              );
            })
          )}
        </div>
      </div>
    </div>
  );
}
