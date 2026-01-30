// lib/google-maps-service.ts
import { retryWithBackoff } from "./retry-utility";

const GOOGLE_MAPS_API_KEY = process.env.NEXT_PUBLIC_GOOGLE_MAPS_API_KEY;

interface Location {
  lat: number;
  lng: number;
}

interface PlaceResult {
  id?: string;
  displayName?: {
    text: string;
    languageCode: string;
  };
  formattedAddress?: string;
  location?: {
    latitude: number;
    longitude: number;
  };
  rating?: number;
  userRatingCount?: number;
  types?: string[];
  regularOpeningHours?: {
    openNow?: boolean;
    weekdayDescriptions?: string[];
  };
  nationalPhoneNumber?: string;
  internationalPhoneNumber?: string;
}

/**
 * Text Search using Places API (New) to find location coordinates
 */
export async function geocodeLocation(place: string): Promise<Location | null> {
  if (!GOOGLE_MAPS_API_KEY) {
    console.error("Google Maps API key not configured");
    return null;
  }

  try {
    const url = "https://places.googleapis.com/v1/places:searchText";

    const response = await retryWithBackoff(
      async () => {
        const res = await fetch(url, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": GOOGLE_MAPS_API_KEY,
            "X-Goog-FieldMask": "places.location,places.displayName",
          },
          body: JSON.stringify({
            textQuery: place,
            maxResultCount: 1,
          }),
          signal: AbortSignal.timeout(8000),
        });

        if (!res.ok) {
          const errorText = await res.text();
          throw new Error(
            `HTTP error! status: ${res.status}, body: ${errorText}`
          );
        }
        return res;
      },
      3,
      1000
    );

    const data = await response.json();

    if (data.places && data.places.length > 0) {
      const place = data.places[0];
      if (place.location) {
        return {
          lat: place.location.latitude,
          lng: place.location.longitude,
        };
      }
    }

    console.warn(`Geocoding failed for "${place}": No results found`);
    return null;
  } catch (error) {
    console.error("Geocoding error:", error);
    return null;
  }
}

/**
 * Parse location input - supports multiple formats:
 * - "lat,lng" (e.g., "40.7128,-74.0060")
 * - "lat lng" (e.g., "40.7128 -74.0060")
 * - Place name (e.g., "New York, NY")
 */
async function parseLocation(
  location: string | Location
): Promise<Location | null> {
  // If already a Location object
  if (typeof location === "object" && "lat" in location && "lng" in location) {
    return location;
  }

  // If string, try to parse
  if (typeof location === "string") {
    const trimmed = location.trim();

    // Try parsing as coordinates first
    const coordParts = trimmed.includes(",")
      ? trimmed.split(",").map((p) => p.trim())
      : trimmed.split(/\s+/);

    if (coordParts.length === 2) {
      try {
        const lat = parseFloat(coordParts[0]);
        const lng = parseFloat(coordParts[1]);

        if (
          !isNaN(lat) &&
          !isNaN(lng) &&
          lat >= -90 &&
          lat <= 90 &&
          lng >= -180 &&
          lng <= 180
        ) {
          return { lat, lng };
        }
      } catch (error) {
        // Not valid coordinates, continue to geocoding
      }
    }

    // Try geocoding the place name using Places API
    return await geocodeLocation(trimmed);
  }

  return null;
}

/**
 * Fetch nearby clinics and hospitals using Places API (New) - Nearby Search
 */
export async function fetchNearbyClinics(
  location: string | Location,
  radius: number = 5000,
  emergencyOnly: boolean = false
): Promise<string> {
  if (!GOOGLE_MAPS_API_KEY) {
    return "Google Maps API key not configured. Please contact support.";
  }

  // Parse the location
  const coords = await parseLocation(location);

  if (!coords) {
    return "Couldn't determine coordinates for that location. Please provide coordinates in format 'lat,lng' (e.g., '40.7128,-74.0060') or a specific place name (e.g., 'Jaffna, Sri Lanka').";
  }

  try {
    const url = "https://places.googleapis.com/v1/places:searchNearby";

    // Choose types based on whether it's an emergency search
    const includedTypes = emergencyOnly
      ? ["hospital", "emergency_room"]
      : ["hospital", "doctor", "pharmacy"];

    const response = await retryWithBackoff(
      async () => {
        const res = await fetch(url, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": GOOGLE_MAPS_API_KEY,
            "X-Goog-FieldMask":
              "places.displayName," +
              "places.formattedAddress," +
              "places.location," +
              "places.rating," +
              "places.userRatingCount," +
              "places.types," +
              "places.regularOpeningHours," +
              "places.nationalPhoneNumber," +
              "places.internationalPhoneNumber",
          },
          body: JSON.stringify({
            includedTypes: includedTypes,
            maxResultCount: 10,
            locationRestriction: {
              circle: {
                center: {
                  latitude: coords.lat,
                  longitude: coords.lng,
                },
                radius: radius,
              },
            },
            rankPreference: "DISTANCE",
          }),
          signal: AbortSignal.timeout(8000),
        });

        if (!res.ok) {
          const errorText = await res.text();
          throw new Error(
            `HTTP error! status: ${res.status}, body: ${errorText}`
          );
        }
        return res;
      },
      3,
      1000
    );

    const data = await response.json();
    const results: PlaceResult[] = data.places || [];

    if (results.length === 0) {
      const facilityType = emergencyOnly
        ? "emergency rooms"
        : "clinics or hospitals";
      return `No ${facilityType} found within ${
        radius / 1000
      }km of your location. Try expanding your search radius or verifying your location.`;
    }

    // Format results professionally
    const clinicList = results.slice(0, 8).map((place, index) => {
      const name = place.displayName?.text || "Unknown";
      const address = place.formattedAddress || "";
      const phone =
        (place as any).nationalPhoneNumber ||
        (place as any).internationalPhoneNumber ||
        "";
      const rating = place.rating
        ? ` - Rating: ${place.rating}/5 (${place.userRatingCount || 0} reviews)`
        : "";
      const openNow = place.regularOpeningHours?.openNow ? " - Open Now" : "";

      let formatted = `${index + 1}. ${name}\n   Location: ${address}`;
      if (phone) {
        formatted += `\n   Phone: ${phone}`;
      }
      formatted += `${rating}${openNow}`;

      return formatted;
    });

    const facilityType = emergencyOnly
      ? "emergency rooms/hospitals"
      : "medical facilities";
    return `Found ${
      results.length
    } nearby ${facilityType}:\n\n${clinicList.join("\n\n")}`;
  } catch (error) {
    console.error("Error fetching nearby clinics:", error);
    return `Error connecting to Google Places API: ${
      error instanceof Error ? error.message : "Unknown error"
    }. Please try again later.`;
  }
}

/**
 * Get distance and duration between two locations using Routes API
 */
export async function getDirections(
  origin: string | Location,
  destination: string | Location
): Promise<string> {
  if (!GOOGLE_MAPS_API_KEY) {
    return "Google Maps API key not configured.";
  }

  try {
    const originCoords = await parseLocation(origin);
    const destCoords = await parseLocation(destination);

    if (!originCoords || !destCoords) {
      return "Could not determine coordinates for the provided locations.";
    }

    const url = "https://routes.googleapis.com/directions/v2:computeRoutes";

    const response = await retryWithBackoff(
      async () => {
        const res = await fetch(url, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": GOOGLE_MAPS_API_KEY,
            "X-Goog-FieldMask": "routes.distanceMeters,routes.duration",
          },
          body: JSON.stringify({
            origin: {
              location: {
                latLng: {
                  latitude: originCoords.lat,
                  longitude: originCoords.lng,
                },
              },
            },
            destination: {
              location: {
                latLng: {
                  latitude: destCoords.lat,
                  longitude: destCoords.lng,
                },
              },
            },
            travelMode: "DRIVE",
            routingPreference: "TRAFFIC_AWARE",
          }),
          signal: AbortSignal.timeout(8000),
        });

        if (!res.ok) {
          const errorText = await res.text();
          throw new Error(
            `HTTP error! status: ${res.status}, body: ${errorText}`
          );
        }
        return res;
      },
      3,
      1000
    );

    const data = await response.json();

    if (data.routes && data.routes.length > 0) {
      const route = data.routes[0];
      const distanceMeters = route.distanceMeters || 0;
      const durationSeconds = route.duration
        ? parseInt(route.duration.replace("s", ""))
        : 0;

      const distanceKm = (distanceMeters / 1000).toFixed(1);
      const durationMin = Math.round(durationSeconds / 60);

      return `Distance: ${distanceKm} km, Estimated travel time: ${durationMin} minutes`;
    }

    return "Could not calculate directions between the specified locations.";
  } catch (error) {
    console.error("Error getting directions:", error);
    return `Error fetching directions: ${
      error instanceof Error ? error.message : "Unknown error"
    }`;
  }
}

/**
 * Helper to validate if Google Maps is configured
 */
export function isGoogleMapsConfigured(): boolean {
  return !!GOOGLE_MAPS_API_KEY && GOOGLE_MAPS_API_KEY.length > 0;
}
