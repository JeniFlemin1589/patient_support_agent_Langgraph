import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

export function middleware(request: NextRequest) {
  const origin = request.headers.get("origin");

  const allowedOrigins = [
    "https://alpha-app-harley.10qbit.com",
    "https://alpha-admin-harley.10qbit.com",
    "http://localhost:3000",
    "http://localhost:3001",
  ];

  const isAllowedOrigin = true; // Temporarily allow all origins to debug

  // Handle preflight requests
  if (request.method === "OPTIONS") {
    const response = NextResponse.json({}, { status: 200 });

    response.headers.set("Access-Control-Allow-Origin", origin || "*");
    response.headers.set(
      "Access-Control-Allow-Methods",
      "GET, POST, PUT, DELETE, PATCH, OPTIONS"
    );
    response.headers.set(
      "Access-Control-Allow-Headers",
      "Content-Type, Authorization, X-Requested-With"
    );
    response.headers.set("Access-Control-Max-Age", "86400");
    response.headers.set("Access-Control-Allow-Credentials", "true");

    return response;
  }

  // For all other requests, add CORS headers to the response
  const response = NextResponse.next();

  response.headers.set("Access-Control-Allow-Origin", origin || "*");
  response.headers.set(
    "Access-Control-Allow-Methods",
    "GET, POST, PUT, DELETE, PATCH, OPTIONS"
  );
  response.headers.set(
    "Access-Control-Allow-Headers",
    "Content-Type, Authorization, X-Requested-With"
  );
  response.headers.set("Access-Control-Allow-Credentials", "true");

  return response;
}

export const config = {
  matcher: ["/api/:path*", "/((?!_next/static|_next/image|favicon.ico).*)"],
};
