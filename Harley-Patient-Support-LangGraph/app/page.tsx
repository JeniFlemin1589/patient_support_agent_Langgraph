// app/page.tsx
"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { ChatInterface } from "@/components/chat-interface";
import { CommunicationsDashboard } from "@/components/communications-dashboard";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { User, Stethoscope, Loader2 } from "lucide-react";

export default function Home() {
  const [patientId, setPatientId] = useState("");
  const [email, setEmail] = useState("");
  const [name, setName] = useState("");
  const [contact, setContact] = useState("");
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [inputEmail, setInputEmail] = useState("");
  const [inputName, setInputName] = useState("");
  const [inputContact, setInputContact] = useState("");
  const [showPortalSelection, setShowPortalSelection] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const router = useRouter();

  const handleLogin = async () => {
    if (!inputEmail.trim() || !inputName.trim() || !inputContact.trim()) {
      alert("Please enter email, name, and contact number");
      return;
    }

    setIsLoading(true);
    try {
      // Store patient profile in MongoDB
      const response = await fetch("/api/patient/profile", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          email: inputEmail,
          name: inputName,
          contact: inputContact,
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to create patient profile");
      }

      const newPatientId = inputEmail.replace(/[^a-zA-Z0-9]/g, "");
      setEmail(inputEmail);
      setName(inputName);
      setContact(inputContact);
      setPatientId(newPatientId);
      setIsLoggedIn(true);
    } catch (error) {
      console.error("Login error:", error);
      alert("Failed to create patient profile. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleDoctorPortal = () => {
    router.push("/doctor");
  };

  if (showPortalSelection) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 via-purple-50 to-pink-50 flex items-center justify-center p-4">
        <div className="max-w-4xl w-full">
          <div className="text-center mb-12">
            <h1 className="text-5xl font-bold text-gray-900 mb-4">
              Harley Health Portal
            </h1>
            <p className="text-xl text-gray-600">
              Select your portal to continue
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-6">
            {/* Patient Portal */}
            <div
              onClick={() => setShowPortalSelection(false)}
              className="bg-white rounded-2xl shadow-xl p-8 hover:shadow-2xl transition-all cursor-pointer transform hover:scale-105 border-2 border-transparent hover:border-blue-500"
            >
              <div className="flex flex-col items-center text-center">
                <div className="w-20 h-20 bg-blue-100 rounded-full flex items-center justify-center mb-6">
                  <User className="h-10 w-10 text-blue-600" />
                </div>
                <h2 className="text-2xl font-bold text-gray-900 mb-3">
                  Patient Portal
                </h2>
                <p className="text-gray-600 mb-6">
                  Access your health records, chat with AI assistant, and manage
                  your healthcare
                </p>
                <div className="w-full">
                  <button className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-6 rounded-lg transition">
                    Enter as Patient
                  </button>
                </div>
              </div>
            </div>

            {/* Doctor Portal */}
            <div
              onClick={handleDoctorPortal}
              className="bg-white rounded-2xl shadow-xl p-8 hover:shadow-2xl transition-all cursor-pointer transform hover:scale-105 border-2 border-transparent hover:border-green-500"
            >
              <div className="flex flex-col items-center text-center">
                <div className="w-20 h-20 bg-green-100 rounded-full flex items-center justify-center mb-6">
                  <Stethoscope className="h-10 w-10 text-green-600" />
                </div>
                <h2 className="text-2xl font-bold text-gray-900 mb-3">
                  Doctor Portal
                </h2>
                <p className="text-gray-600 mb-6">
                  Access patient communications, review cases, and manage
                  healthcare delivery
                </p>
                <div className="w-full">
                  <button className="w-full bg-green-600 hover:bg-green-700 text-white font-semibold py-3 px-6 rounded-lg transition">
                    Enter as Doctor
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (!isLoggedIn) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-blue-100 flex items-center justify-center p-4">
        <div className="bg-white rounded-lg shadow-lg p-8 max-w-md w-full">
          <button
            onClick={() => setShowPortalSelection(true)}
            className="text-sm text-blue-600 hover:text-blue-700 mb-4"
          >
            ← Back to portal selection
          </button>

          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            Harley Health
          </h1>
          <p className="text-gray-600 mb-6">
            Your intelligent healthcare portal
          </p>

          <div className="space-y-4">
            <input
              type="text"
              value={inputName}
              onChange={(e) => setInputName(e.target.value)}
              placeholder="Enter your full name"
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-600 focus:border-transparent outline-none"
              disabled={isLoading}
            />

            <input
              type="tel"
              value={inputContact}
              onChange={(e) => setInputContact(e.target.value)}
              placeholder="Enter your contact number"
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-600 focus:border-transparent outline-none"
              disabled={isLoading}
            />

            <input
              type="email"
              value={inputEmail}
              onChange={(e) => setInputEmail(e.target.value)}
              placeholder="Enter your email"
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-600 focus:border-transparent outline-none"
              disabled={isLoading}
              onKeyPress={(e) => e.key === "Enter" && handleLogin()}
            />

            <button
              onClick={handleLogin}
              disabled={isLoading}
              className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white font-semibold py-2 px-4 rounded-lg transition flex items-center justify-center gap-2"
            >
              {isLoading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Creating Profile...
                </>
              ) : (
                "Enter Portal"
              )}
            </button>
          </div>

          <p className="text-xs text-gray-500 text-center mt-4">
            Demo mode - Enter your details to continue
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-6xl mx-auto">
        <div className="mb-8">
          <button
            onClick={() => {
              setIsLoggedIn(false);
              setShowPortalSelection(true);
            }}
            className="text-sm text-blue-600 hover:text-blue-700 mb-4"
          >
            ← Back to portal selection
          </button>
          <h1 className="text-4xl font-bold text-gray-900">
            Welcome, {name || email}
          </h1>
          <p className="text-gray-600 mt-2">
            Your healthcare assistant is ready to help
          </p>
        </div>

        <Tabs defaultValue="chat" className="w-full">
          <TabsList className="grid w-full grid-cols-2 mb-6">
            <TabsTrigger value="chat">Chat Assistant</TabsTrigger>
            <TabsTrigger value="communications">Communications</TabsTrigger>
          </TabsList>

          <TabsContent value="chat" className="space-y-4">
            <ChatInterface
              patientId={patientId}
              email={email}
              name={name}
              contact={contact}
            />
          </TabsContent>

          <TabsContent value="communications" className="space-y-4">
            <CommunicationsDashboard patientId={patientId} />
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
