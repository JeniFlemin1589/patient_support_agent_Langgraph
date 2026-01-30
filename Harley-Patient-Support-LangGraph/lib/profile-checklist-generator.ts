// lib/profile-checklist-generator.ts (FIXED - Client-side only)
/**
 * Enhanced checklist item with profile comparison
 */
export interface ProfileChecklistItem {
  id: string;
  label: string;
  status: "present" | "missing" | "updated" | "partial";
  checked: boolean;
  category: string;
  importance: "critical" | "high" | "medium" | "low";
  profileValue?: string;
  summaryValue?: string;
  notes?: string;
}

/**
 * Generate profile-based checklist by calling API
 */
export async function generateProfileChecklist(
  summary: string,
  patientId: string,
  patientEmail?: string
): Promise<ProfileChecklistItem[]> {
  console.log("[Profile Checklist] Calling API to generate checklist");

  try {
    const response = await fetch("/api/profile-checklist", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        summary,
        patientId,
        patientEmail,
      }),
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    const data = await response.json();

    if (data.error) {
      throw new Error(data.error);
    }

    console.log(
      "[Profile Checklist] Generated",
      data.items.length,
      "items",
      data.hasProfile ? "with profile" : "without profile"
    );

    return data.items;
  } catch (error) {
    console.error("[Profile Checklist] Error calling API:", error);
    // Return basic fallback
    return generateBasicFallback(summary);
  }
}

/**
 * Basic fallback if API fails
 */
function generateBasicFallback(summary: string): ProfileChecklistItem[] {
  const items: ProfileChecklistItem[] = [];
  const summaryLower = summary.toLowerCase();

  if (summaryLower.includes("age") || summaryLower.includes("years old")) {
    items.push({
      id: `fallback-age-${Date.now()}`,
      label: "Patient age documented",
      status: "present",
      checked: true,
      category: "Demographics & Basic Info",
      importance: "critical",
      notes: "Age information found in summary",
    });
  }

  if (summaryLower.includes("weight") || summaryLower.includes("kg")) {
    items.push({
      id: `fallback-weight-${Date.now()}`,
      label: "Current weight documented",
      status: "present",
      checked: true,
      category: "Weight Management History",
      importance: "high",
      notes: "Weight information found in summary",
    });
  }

  if (summaryLower.includes("medication") || summaryLower.includes("drug")) {
    items.push({
      id: `fallback-meds-${Date.now()}`,
      label: "Medications discussed",
      status: "present",
      checked: true,
      category: "Current Medications",
      importance: "critical",
      notes: "Medication information found in summary",
    });
  }

  return items;
}

/**
 * Get status color for UI
 */
export function getStatusColor(status: ProfileChecklistItem["status"]): string {
  switch (status) {
    case "present":
      return "text-green-600 bg-green-50 border-green-200";
    case "updated":
      return "text-blue-600 bg-blue-50 border-blue-200";
    case "missing":
      return "text-red-600 bg-red-50 border-red-200";
    case "partial":
      return "text-yellow-600 bg-yellow-50 border-yellow-200";
  }
}

/**
 * Get status icon for UI
 */
export function getStatusIcon(status: ProfileChecklistItem["status"]): string {
  switch (status) {
    case "present":
      return "✓";
    case "updated":
      return "↻";
    case "missing":
      return "✗";
    case "partial":
      return "◐";
  }
}

/**
 * Group checklist by category
 */
export function groupProfileChecklistByCategory(
  checklist: ProfileChecklistItem[]
): Record<string, ProfileChecklistItem[]> {
  const grouped: Record<string, ProfileChecklistItem[]> = {};

  checklist.forEach((item) => {
    if (!grouped[item.category]) {
      grouped[item.category] = [];
    }
    grouped[item.category].push(item);
  });

  return grouped;
}

/**
 * Calculate profile comparison statistics
 */
export function getProfileComparisonStats(checklist: ProfileChecklistItem[]) {
  const total = checklist.length;
  const present = checklist.filter((i) => i.status === "present").length;
  const updated = checklist.filter((i) => i.status === "updated").length;
  const missing = checklist.filter((i) => i.status === "missing").length;
  const partial = checklist.filter((i) => i.status === "partial").length;

  return {
    total,
    present,
    updated,
    missing,
    partial,
    completeness: Math.round(((present + updated) / total) * 100),
  };
}
