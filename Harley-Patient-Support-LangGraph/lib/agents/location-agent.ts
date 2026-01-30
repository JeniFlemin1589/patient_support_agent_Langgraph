// lib/agents/location-agent.ts
import { GoogleGenerativeAI } from "@google/generative-ai";
import type { ChatState } from "../types";
import { retryWithBackoff } from "../retry-utility";
import {
  fetchNearbyClinics,
  geocodeLocation,
  getDirections,
  isGoogleMapsConfigured,
} from "../google-maps-service";

const genai = new GoogleGenerativeAI(process.env.NEXT_PUBLIC_GEMINI_API_KEY!);
const model = genai.getGenerativeModel({ model: "gemini-2.5-flash-lite" });

/**
 * Remove markdown formatting from text
 */
function cleanMarkdown(text: string): string {
  return (
    text
      // Remove bold (**text** or __text__)
      .replace(/(\*\*|__)(.*?)\1/g, "$2")
      // Remove italic (*text* or _text_)
      .replace(/([*_])(.*?)\1/g, "$2")
      // Remove strikethrough (~~text~~)
      .replace(/~~(.*?)~~/g, "$1")
      // Remove inline code (`code`)
      .replace(/`([^`]+)`/g, "$1")
      // Remove headers (# ## ### etc)
      .replace(/^#{1,6}\s+/gm, "")
      // Remove horizontal rules (---, ___, ***)
      .replace(/^[-*_]{3,}\s*$/gm, "")
      // Remove blockquotes (> text)
      .replace(/^>\s+/gm, "")
      // Remove list markers (-, *, +, 1.)
      .replace(/^[\s]*[-*+]\s+/gm, "")
      .replace(/^[\s]*\d+\.\s+/gm, "")
      // Remove links but keep text [text](url) -> text
      .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
      // Remove images ![alt](url)
      .replace(/!\[([^\]]*)\]\([^)]+\)/g, "$1")
      // Clean up multiple spaces
      .replace(/\s+/g, " ")
      .trim()
  );
}

export async function locationAgent(state: ChatState): Promise<{
  answer: string;
  clinicLocations?: string;
  needsLocation?: boolean;
}> {
  const query = state.query.toLowerCase();

  // Check if Google Maps is configured
  if (!isGoogleMapsConfigured()) {
    return {
      answer:
        "Location services are currently unavailable. Please contact support or try again later.",
      needsLocation: false,
    };
  }

  // Check if user is specifically looking for emergency rooms (NOT general emergency)
  const isEmergencyRoomSearch =
    (query.includes("emergency room") ||
      query.includes("emergency rooms") ||
      query.includes("er near") ||
      query.includes("find emergency room")) &&
    query.includes("near");

  // Extract location from query
  const extractLocationPrompt = `Extract the location from this query. If no specific location is mentioned, respond with "NONE".

Query: "${state.query}"

Examples:
"Find clinics near New York" -> New York
"Hospitals in Jaffna, Sri Lanka" -> Jaffna, Sri Lanka
"Urgent care centers around 40.7128,-74.0060" -> 40.7128,-74.0060
"Where can I find a doctor?" -> NONE

Respond with ONLY the location or "NONE":`;

  try {
    const locationResponse = await retryWithBackoff(
      async () => {
        return await model.generateContent(extractLocationPrompt);
      },
      2,
      1000,
    );

    const extractedLocation = locationResponse.response.text().trim();

    // If no location found, ask user for location
    if (extractedLocation === "NONE" || !extractedLocation) {
      return {
        answer:
          "To find nearby clinics and hospitals, I need your location. Please provide:\n\n1. A city/area name (e.g., 'New York, NY' or 'Jaffna, Sri Lanka')\n2. Your coordinates (e.g., '40.7128,-74.0060')\n3. Or ask 'find clinics near [location]'",
        needsLocation: true,
      };
    }

    // Fetch nearby facilities based on search type
    let clinicInfo: string;
    try {
      if (isEmergencyRoomSearch) {
        // Only emergency rooms with phone numbers
        clinicInfo = await fetchNearbyClinics(extractedLocation, 5000, true);
      } else {
        // All medical facilities (hospitals, clinics, doctors, pharmacies)
        clinicInfo = await fetchNearbyClinics(extractedLocation, 5000, false);
      }
    } catch (error) {
      console.error("Error fetching clinics:", error);
      return {
        answer: `I found your location (${extractedLocation}), but encountered an error fetching nearby medical facilities. Please try again or verify your location.`,
        needsLocation: false,
      };
    }

    // Generate a concise introduction
    const facilityType = isEmergencyRoomSearch
      ? "emergency rooms with contact numbers"
      : "medical facilities including hospitals, clinics, and pharmacies";
    const responsePrompt = `You are a helpful healthcare location assistant. Write a brief, friendly introduction (2-3 sentences max) for the nearby ${facilityType} search.

User Query: "${state.query}"
Location: ${extractedLocation}
Search Type: ${
      isEmergencyRoomSearch ? "Emergency Rooms Only" : "All Medical Facilities"
    }

Create a SHORT introduction that:
1. Confirms the location
2. States the type of facilities found
3. ${
      isEmergencyRoomSearch
        ? "Emphasizes calling the numbers provided before visiting"
        : "Mentions facilities include hospitals, clinics, doctors, and pharmacies"
    }

IMPORTANT: Write in plain text without any markdown formatting. Keep it very brief - just 2-3 sentences. Do not list the facilities, just introduce them.`;

    const response = await retryWithBackoff(
      async () => {
        return await model.generateContent(responsePrompt);
      },
      3,
      1000,
    );

    const rawIntro = response.response.text();
    const cleanIntro = cleanMarkdown(rawIntro);

    // Different notes based on search type
    const closingNote = isEmergencyRoomSearch
      ? "\n\nIMPORTANT: Please call the emergency room numbers listed above before visiting to confirm availability and services. For life-threatening emergencies, call 999 or your local emergency number immediately."
      : "\n\nNote: This list includes hospitals, clinics, doctor offices, and pharmacies. Consider factors like ratings, distance, services offered, and whether the facility is currently open. For life-threatening emergencies, call 999 immediately.";

    const finalAnswer = `${cleanIntro}\n\n${clinicInfo}${closingNote}`;

    return {
      answer: finalAnswer,
      clinicLocations: clinicInfo,
      needsLocation: false,
    };
  } catch (error) {
    console.error("Location agent error:", error);

    return {
      answer:
        "I'm having trouble processing location information right now. For immediate assistance, please:\n\n1. Call your local healthcare provider\n2. Search online for '[your area] clinics'\n3. Use Google Maps to search for 'hospitals near me'\n4. In emergencies, call 999 or your local emergency number",
      needsLocation: false,
    };
  }
}
