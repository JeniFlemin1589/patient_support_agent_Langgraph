// lib/ai-checklist-generator.ts
import { GoogleGenerativeAI } from "@google/generative-ai";
import { retryWithBackoff } from "./retry-utility";

const genai = new GoogleGenerativeAI(process.env.NEXT_PUBLIC_GEMINI_API_KEY!);
const model = genai.getGenerativeModel({ model: "gemini-2.5-flash-lite" });

/**
 * Interface for AI-generated checklist items
 */
export interface AIChecklistItem {
  id: string;
  label: string;
  checked: boolean;
  category: string;
  importance: "critical" | "high" | "medium" | "low";
}

/**
 * Interface for checklist generation response
 */
interface ChecklistGenerationResponse {
  items: Array<{
    label: string;
    category: string;
    checked: boolean;
    importance: "critical" | "high" | "medium" | "low";
  }>;
  overallCompleteness: number;
  qualityScore: number;
  missingElements: string[];
}

/**
 * Generate checklist using AI based on medical summary
 */
export async function generateAIChecklist(
  summary: string,
): Promise<AIChecklistItem[]> {
  if (!summary || summary.trim().length === 0) {
    console.warn("[AI Checklist] Empty summary provided");
    return [];
  }

  const prompt = `You are a medical documentation quality analyst. Analyze this patient conversation summary and create a comprehensive documentation checklist.

MEDICAL SUMMARY:
"""
${summary}
"""

Your task is to evaluate what medical information IS present in this summary and create checklist items accordingly.

CHECKLIST CATEGORIES (Use these exact names):
1. "Chief Complaint & Symptoms"
2. "Medical History"
3. "Clinical Assessment"
4. "Vital Signs & Measurements"
5. "Medications & Treatments"
6. "Diagnostic Plans"
7. "Recommendations & Follow-up"
8. "Patient Education"
9. "Documentation Quality"

For each piece of information found in the summary, create a checklist item with:
- label: Clear description of what was documented
- category: One of the 9 categories above
- checked: true (since it's present in summary)
- importance: "critical" | "high" | "medium" | "low"

IMPORTANCE LEVELS:
- critical: Life-critical info (chest pain severity, breathing issues, allergies)
- high: Essential medical info (diagnosis, vital signs, medications)
- medium: Important context (medical history, timeline, patient concerns)
- low: Supplementary info (general recommendations, educational content)

Also identify any MISSING critical elements that should have been documented but weren't.

Respond ONLY with valid JSON in this exact format:
{
  "items": [
    {
      "label": "Patient's chief complaint documented (chest pain)",
      "category": "Chief Complaint & Symptoms",
      "checked": true,
      "importance": "critical"
    }
  ],
  "overallCompleteness": 85,
  "qualityScore": 90,
  "missingElements": ["Blood pressure measurement", "Family history"]
}

IMPORTANT RULES:
1. Create 8-15 checklist items (not too few, not too many)
2. Only mark items as checked:true if that information IS in the summary
3. Be specific in labels (mention actual symptoms/conditions found)
4. overallCompleteness: 0-100 (how complete is documentation)
5. qualityScore: 0-100 (overall documentation quality)
6. missingElements: Array of critical info that should have been included
7. Respond with ONLY the JSON object, no markdown, no explanations`;

  try {
    console.log("[AI Checklist] Generating checklist for summary...");

    const response = await retryWithBackoff(
      async () => {
        return await model.generateContent(prompt);
      },
      3,
      1000,
    );

    let responseText = response.response.text().trim();

    // Remove markdown code blocks if present
    responseText = responseText
      .replace(/```json\n?/g, "")
      .replace(/```\n?/g, "");

    console.log(
      "[AI Checklist] Raw AI response:",
      responseText.substring(0, 200),
    );

    const parsedResponse: ChecklistGenerationResponse =
      JSON.parse(responseText);

    // Validate response structure
    if (!parsedResponse.items || !Array.isArray(parsedResponse.items)) {
      throw new Error("Invalid AI response: missing items array");
    }

    // Convert to AIChecklistItem format with unique IDs
    const checklistItems: AIChecklistItem[] = parsedResponse.items.map(
      (item, index) => ({
        id: `ai-item-${Date.now()}-${index}`,
        label: item.label,
        category: item.category,
        checked: item.checked,
        importance: item.importance,
      }),
    );

    // Add quality metadata items
    if (
      parsedResponse.missingElements &&
      parsedResponse.missingElements.length > 0
    ) {
      checklistItems.push({
        id: `ai-item-missing-${Date.now()}`,
        label: `Missing elements identified: ${parsedResponse.missingElements.join(
          ", ",
        )}`,
        category: "Documentation Quality",
        checked: false,
        importance: "high",
      });
    }

    checklistItems.push({
      id: `ai-item-completeness-${Date.now()}`,
      label: `Overall documentation completeness: ${parsedResponse.overallCompleteness}%`,
      category: "Documentation Quality",
      checked: parsedResponse.overallCompleteness >= 80,
      importance: "medium",
    });

    checklistItems.push({
      id: `ai-item-quality-${Date.now()}`,
      label: `Documentation quality score: ${parsedResponse.qualityScore}/100`,
      category: "Documentation Quality",
      checked: parsedResponse.qualityScore >= 80,
      importance: "medium",
    });

    console.log(
      "[AI Checklist] Successfully generated",
      checklistItems.length,
      "items",
    );

    return checklistItems;
  } catch (error) {
    console.error("[AI Checklist] Error generating AI checklist:", error);

    // Fallback to basic analysis
    return generateFallbackChecklist(summary);
  }
}

/**
 * Fallback checklist generation if AI fails
 */
function generateFallbackChecklist(summary: string): AIChecklistItem[] {
  console.warn("[AI Checklist] Using fallback checklist generation");

  const items: AIChecklistItem[] = [];
  const wordCount = summary.split(/\s+/).length;
  const hasStructure = summary.includes("\n\n") || summary.includes(":");
  const lines = summary.split("\n").filter((line) => line.trim());

  // Basic checks
  items.push({
    id: `fallback-1-${Date.now()}`,
    label: `Conversation summary created (${wordCount} words)`,
    category: "Documentation Quality",
    checked: wordCount >= 50,
    importance: "medium",
  });

  items.push({
    id: `fallback-2-${Date.now()}`,
    label: "Summary has structured format",
    category: "Documentation Quality",
    checked: hasStructure,
    importance: "low",
  });

  items.push({
    id: `fallback-3-${Date.now()}`,
    label: `Multiple data points captured (${lines.length} sections)`,
    category: "Documentation Quality",
    checked: lines.length >= 3,
    importance: "medium",
  });

  // Keyword-based detection
  const summaryLower = summary.toLowerCase();

  if (summaryLower.includes("pain") || summaryLower.includes("symptom")) {
    items.push({
      id: `fallback-symptoms-${Date.now()}`,
      label: "Patient symptoms documented",
      category: "Chief Complaint & Symptoms",
      checked: true,
      importance: "high",
    });
  }

  if (summaryLower.includes("history") || summaryLower.includes("previous")) {
    items.push({
      id: `fallback-history-${Date.now()}`,
      label: "Medical history reviewed",
      category: "Medical History",
      checked: true,
      importance: "medium",
    });
  }

  if (summaryLower.includes("recommend") || summaryLower.includes("should")) {
    items.push({
      id: `fallback-recs-${Date.now()}`,
      label: "Medical recommendations provided",
      category: "Recommendations & Follow-up",
      checked: true,
      importance: "high",
    });
  }

  console.log("[AI Checklist] Fallback generated", items.length, "items");

  return items;
}

/**
 * Group checklist items by category
 */
export function groupChecklistByCategory(
  checklist: AIChecklistItem[],
): Record<string, AIChecklistItem[]> {
  const grouped: Record<string, AIChecklistItem[]> = {};

  checklist.forEach((item) => {
    if (!grouped[item.category]) {
      grouped[item.category] = [];
    }
    grouped[item.category].push(item);
  });

  return grouped;
}

/**
 * Calculate checklist completion percentage
 */
export function getChecklistCompletionPercentage(
  checklist: AIChecklistItem[],
): number {
  if (checklist.length === 0) return 0;

  const checkedCount = checklist.filter((item) => item.checked).length;
  return Math.round((checkedCount / checklist.length) * 100);
}

/**
 * Get summary statistics
 */
export function getSummaryStatistics(summary: string) {
  const wordCount = summary.split(/\s+/).length;
  const lineCount = summary.split("\n").filter((line) => line.trim()).length;
  const hasStructuredFormat = summary.includes("\n\n") || summary.includes(":");

  return {
    wordCount,
    lineCount,
    hasStructuredFormat,
    estimatedReadingTime: Math.ceil(wordCount / 200),
  };
}

/**
 * Get importance color for UI
 */
export function getImportanceColor(
  importance: AIChecklistItem["importance"],
): string {
  switch (importance) {
    case "critical":
      return "text-red-600 bg-red-50 border-red-200";
    case "high":
      return "text-orange-600 bg-orange-50 border-orange-200";
    case "medium":
      return "text-blue-600 bg-blue-50 border-blue-200";
    case "low":
      return "text-gray-600 bg-gray-50 border-gray-200";
  }
}
