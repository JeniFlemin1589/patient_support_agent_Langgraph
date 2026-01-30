"use client";

import { useState, useRef, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Send,
  Loader2,
  RefreshCw,
  CheckCircle,
  AlertCircle,
  LogOut,
  Clock,
} from "lucide-react";

interface Message {
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: Date;
  agentType?: string;
  emergencyNumber?: string;
  nearbyClinicLocations?: string[];
  needsLocation?: boolean;
  severity?: string;
}

interface ChatInterfaceProps {
  patientId: string;
  email: string;
  name?: string;
  contact?: string;
}

const INACTIVITY_TIMEOUT = 40000; // 40 seconds
const AUTO_END_TIMEOUT = 15000; // 15 seconds after prompt to auto-end
const INITIAL_CHECKPOINT = 6; // First checkpoint at 6 Q&A pairs
const EXTENDED_CHECKPOINT = 3; // Extended checkpoint at 3 more Q&A pairs

export function ChatInterface({ patientId, email }: ChatInterfaceProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [showWelcome, setShowWelcome] = useState(true);
  const [summaryCreated, setSummaryCreated] = useState(false);
  const [creatingAutoSummary, setCreatingAutoSummary] = useState(false);
  const [sessionId, setSessionId] = useState<string>("");
  const [qaPairCount, setQaPairCount] = useState(0);
  const [displayedSummary, setDisplayedSummary] = useState<string>("");
  const [showCheckpoint, setShowCheckpoint] = useState(false);
  const [conversationLoop, setConversationLoop] = useState(1);
  const [showEndMessage, setShowEndMessage] = useState(false);
  const [currentLoopStartIndex, setCurrentLoopStartIndex] = useState(0);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [error, setError] = useState<string | null>(null);
  const [showThankYou, setShowThankYou] = useState(false);
  const [conversationEnded, setConversationEnded] = useState(false);
  const [showInactivityPrompt, setShowInactivityPrompt] = useState(false);
  const [inactivityTimer, setInactivityTimer] = useState<NodeJS.Timeout | null>(
    null,
  );
  const [checkpointType, setCheckpointType] = useState<
    "regular" | "extended" | "inactivity"
  >("regular");
  const [autoEndTimer, setAutoEndTimer] = useState<NodeJS.Timeout | null>(null);

  const storageKey = `chat_messages_${patientId}`;
  const sessionStorageKey = `session_id_${patientId}`;

  // Reset inactivity timer


  const resetInactivityTimer = () => {
    if (inactivityTimer) clearTimeout(inactivityTimer);
    if (autoEndTimer) clearTimeout(autoEndTimer);

    // Don't set timer if conversation ended or checkpoint is showing or already prompting
    if (conversationEnded || showCheckpoint || showInactivityPrompt) {
      return;
    }

    const timer = setTimeout(() => {
      if (messages.length > 0 && !conversationEnded && !showCheckpoint) {
        console.log("[Chat] Inactivity detected - showing end session prompt");
        setCheckpointType("inactivity");
        setShowInactivityPrompt(true);
      }
    }, INACTIVITY_TIMEOUT);

    setInactivityTimer(timer);
  };

  // Cleanup timer on unmount
  useEffect(() => {
    return () => {
      if (inactivityTimer) clearTimeout(inactivityTimer);
      if (autoEndTimer) clearTimeout(autoEndTimer);
    };
  }, [inactivityTimer, autoEndTimer]);

  // Reset timer when user types
  useEffect(() => {
    if (input.length > 0) {
      resetInactivityTimer();
    }
  }, [input]);

  useEffect(() => {
    const loadMessages = () => {
      try {
        const stored = sessionStorage.getItem(storageKey);
        let currentSessionId = sessionStorage.getItem(sessionStorageKey);
        if (!currentSessionId) {
          currentSessionId = `session_${Date.now()}_${Math.random()
            .toString(36)
            .substr(2, 9)}`;
          sessionStorage.setItem(sessionStorageKey, currentSessionId);
        }
        setSessionId(currentSessionId);

        if (stored) {
          const parsed = JSON.parse(stored);
          const messagesWithDates = parsed.map((msg: any) => ({
            ...msg,
            timestamp: new Date(msg.timestamp),
          }));
          setMessages(messagesWithDates);
          setShowWelcome(messagesWithDates.length === 0);
          const userMessages = messagesWithDates.filter(
            (m: Message) => m.role === "user",
          ).length;
          setQaPairCount(userMessages);
        }
      } catch (error) {
        console.error("Error loading messages:", error);
      }
    };

    loadMessages();
  }, [patientId, storageKey, sessionStorageKey]);

  useEffect(() => {
    if (messages.length > 0) {
      try {
        sessionStorage.setItem(storageKey, JSON.stringify(messages));
      } catch (error) {
        console.error("Error saving messages:", error);
      }
    }
  }, [messages, storageKey]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleClearChat = () => {
    if (window.confirm("Are you sure you want to clear the chat history?")) {
      setMessages([]);
      setShowWelcome(true);
      setSummaryCreated(false);
      setDisplayedSummary("");
      setQaPairCount(0);
      setShowCheckpoint(false);
      setConversationLoop(1);
      setShowEndMessage(false);
      setShowThankYou(false);
      setShowInactivityPrompt(false);
      setConversationEnded(false);
      sessionStorage.removeItem(storageKey);
      sessionStorage.removeItem(sessionStorageKey);
      const newSessionId = `session_${Date.now()}_${Math.random()
        .toString(36)
        .substr(2, 9)}`;
      sessionStorage.setItem(sessionStorageKey, newSessionId);
      setSessionId(newSessionId);
      if (inactivityTimer) {
        clearTimeout(inactivityTimer);
      }
    }
  };

  const createAutoSummary = async (
    messagesToSummarize: Array<{
      role: "user" | "assistant";
      content: string;
      timestamp: string;
    }>,
    agentType: string,
    severity?: string,
  ): Promise<boolean> => {
    try {
      console.log("[v0] ===== CREATING SUMMARY =====");
      console.log("[v0] Messages to summarize:", messagesToSummarize.length);
      console.log("[v0] Agent type:", agentType);
      console.log("[v0] Severity:", severity);
      console.log("[v0] Session ID:", sessionId);
      console.log("[v0] Will store in chat_history collection");

      if (!messagesToSummarize || messagesToSummarize.length === 0) {
        console.error("[v0] No messages to summarize!");
        return false;
      }

      const messagesToSend: Array<{
        role: "user" | "assistant";
        content: string;
        timestamp: string;
      }> = messagesToSummarize.map((msg) => ({
        role: msg.role,
        content: msg.content,
        timestamp: msg.timestamp,
      }));

      const requestBody = {
        patientId,
        email,
        messages: messagesToSend,
        agentType,
        sessionId: sessionId,
        qaPairCount: qaPairCount,
        isConversationComplete: true,
        severity: severity || "medium",
      };

      console.log(
        "[v0] Sending POST request to /api/communications/summary...",
      );
      console.log("[v0] Request body:", JSON.stringify(requestBody, null, 2));

      const response = await fetch("/api/communications/summary", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody),
      });

      console.log(
        "[v0] Response status:",
        response.status,
        response.statusText,
      );

      if (response.ok) {
        const data = await response.json();
        console.log(
          "[v0] ✓ Summary created and stored in chat_history successfully!",
        );
        console.log("[v0] Response data:", data);
        return true;
      } else {
        const errorText = await response.text();
        console.error(
          "[v0] ✗ Failed to create summary - Status:",
          response.status,
        );
        console.error("[v0] Error response:", errorText);
        return false;
      }
    } catch (error) {
      console.error("[v0] ✗ Error creating auto summary:", error);
      if (error instanceof Error) {
        console.error("[v0] Error message:", error.message);
        console.error("[v0] Error stack:", error.stack);
      }
      return false;
    }
  };

  const handleEndConversation = async () => {
    console.log("[v0] ===== HANDLE END CONVERSATION CALLED =====");
    console.log("[v0] Current messages count:", messages.length);
    console.log("[v0] Session ID:", sessionId);

    setShowCheckpoint(false);
    setShowInactivityPrompt(false);
    if (inactivityTimer) {
      clearTimeout(inactivityTimer);
    }

    const allSessionMessages = messages.filter(
      (msg) => msg.role === "user" || msg.role === "assistant",
    );

    console.log("[v0] Filtered session messages:", allSessionMessages.length);

    if (allSessionMessages.length === 0) {
      console.log("[v0] No messages to summarize, skipping summary creation");
      return;
    }

    const messagesToSummarize: Array<{
      role: "user" | "assistant";
      content: string;
      timestamp: string;
    }> = allSessionMessages.map((msg) => ({
      role: msg.role as "user" | "assistant",
      content: msg.content,
      timestamp:
        msg.timestamp instanceof Date
          ? msg.timestamp.toISOString()
          : String(msg.timestamp),
    }));

    // Find the last assistant message to determine severity/agent type
    const lastAssistantMessage = [...messages].reverse().find(
      (m) => m.role === "assistant"
    );

    // Default to stored severity or extract from last message
    // Priority: Emergency agent type -> Critical/High
    let currentSeverity = "medium";
    let currentAgentType = "clinical";

    if (lastAssistantMessage) {
      if (lastAssistantMessage.agentType === "emergency") {
        currentSeverity = "critical";
        currentAgentType = "emergency";
      } else if (lastAssistantMessage.severity) {
        currentSeverity = lastAssistantMessage.severity;
        currentAgentType = lastAssistantMessage.agentType || "clinical";
      }
    }

    console.log(
      "[v0] Prepared",
      messagesToSummarize.length,
      "messages for summary. Severity:",
      currentSeverity
    );

    // Show loading state while confirming/sending
    setLoading(true);

    console.log("[v0] Calling createAutoSummary...");
    const success = await createAutoSummary(
      messagesToSummarize,
      currentAgentType,
      currentSeverity
    );
    console.log("[v0] createAutoSummary completed, success:", success);

    setLoading(false);

    if (success) {
      const summaryMessage: Message = {
        role: "assistant",
        content:
          "I have successfully sent the comprehensive summary to your doctor. Thank you for your time. Goodbye!",
        timestamp: new Date(),
        agentType: "system",
      };
      setMessages((prevMessages) => [...prevMessages, summaryMessage]);

      // Set conversation ended immediately to disable input
      setConversationEnded(true);
      setShowEndMessage(true);
    } else {
      // Fallback message if summary failed (optional, but good UX)
      const errorMessage: Message = {
        role: "assistant",
        content: "There was an issue sending the summary, but your session is saved. Goodbye!",
        timestamp: new Date(),
        agentType: "system",
      };
      setMessages((prevMessages) => [...prevMessages, errorMessage]);
      setConversationEnded(true);
    }
  };

  // Handle Auto-End logic (effect) when Inactivity Prompt is shown
  useEffect(() => {
    if (showInactivityPrompt && !conversationEnded) {
      console.log("[Chat] Inactivity prompt shown. Starting Auto-End timer...");
      const timer = setTimeout(() => {
        console.log("[Chat] Auto-End timer fired. Ending conversation.");
        handleEndConversation();
      }, AUTO_END_TIMEOUT);
      setAutoEndTimer(timer);
    } else {
      if (autoEndTimer) clearTimeout(autoEndTimer);
      setAutoEndTimer(null);
    }
  }, [showInactivityPrompt, conversationEnded]);

  const handleCheckpointContinue = () => {
    console.log("[v0] User chose to continue conversation");
    setShowCheckpoint(false);
    setShowInactivityPrompt(false);
    setConversationLoop(conversationLoop + 1);
    setQaPairCount(0); // Reset Q/A pair count for the new loop
    setCurrentLoopStartIndex(messages.length);
    setCheckpointType("regular"); // Reset to regular for next checkpoint
    resetInactivityTimer(); // Restart inactivity timer
    console.log("[v0] Starting new loop:", {
      newLoop: conversationLoop + 1,
      newStartIndex: messages.length,
    });
  };

  const handleCheckpointEnd = async () => {
    console.log("[v0] User chose to end conversation at checkpoint");
    setShowCheckpoint(false);
    setShowInactivityPrompt(false);
    await handleEndConversation();
  };

  const handleInactivityResponse = async (shouldEnd: boolean) => {
    if (shouldEnd) {
      await handleCheckpointEnd();
    } else {
      setShowInactivityPrompt(false);
      resetInactivityTimer();
    }
  };

  const handleSendMessage = async () => {
    if (!input.trim()) return;

    const currentInput = input;

    // Check if user is responding to checkpoint/inactivity with "yes" or summary keywords
    if (showCheckpoint || showInactivityPrompt) {
      const affirmativeKeywords = [
        "yes",
        "yeah",
        "yep",
        "sure",
        "ok",
        "okay",
        "create summary",
        "end session",
        "end conversation",
        "finish",
        "done",
      ];

      const isAffirmativeForEnd = affirmativeKeywords.some(
        (keyword) =>
          currentInput.toLowerCase().trim() === keyword ||
          currentInput.toLowerCase().includes(keyword),
      );

      if (isAffirmativeForEnd) {
        console.log(
          "[v0] User confirmed end session at checkpoint/inactivity prompt",
        );
        setInput("");

        const userMessage: Message = {
          role: "user",
          content: currentInput,
          timestamp: new Date(),
        };
        setMessages([...messages, userMessage]);

        await handleEndConversation();
        return;
      }

      // If they typed something else at checkpoint, treat as "continue"
      if (showCheckpoint) {
        console.log("[v0] User wants to continue from checkpoint");
        handleCheckpointContinue();
        // Don't return - let the message be processed normally below
      } else if (showInactivityPrompt) {
        console.log("[v0] User active again after inactivity");
        handleInactivityResponse(false);
        // Don't return - let the message be processed normally below
      }
    }

    const endConversationKeywords = [
      "end conversation",
      "end the conversation",
      "finish conversation",
      "stop conversation",
      "create summary",
      "end chat",
      "finish chat",
      "that's all",
      "thats all",
      "i'm done",
      "im done",
      "goodbye",
      "bye",
      "end",
      "finish",
      "done",
      "endconversation",
      "yes end",
      "yes, end",
    ];

    const isEndingConversation = endConversationKeywords.some(
      (keyword) =>
        currentInput.toLowerCase().trim() === keyword ||
        currentInput.toLowerCase().includes(keyword),
    );

    if (isEndingConversation && messages.length > 0) {
      console.log(
        "[v0] User typed end conversation keyword, creating summary and storing in chat_history...",
      );
      setInput("");

      const userMessage: Message = {
        role: "user",
        content: currentInput,
        timestamp: new Date(),
      };
      setMessages([...messages, userMessage]);

      await handleEndConversation();
      return;
    }

    setInput("");
    setLoading(true);
    setError(null);

    const userMessage: Message = {
      role: "user",
      content: currentInput,
      timestamp: new Date(),
    };

    const newMessages = [...messages, userMessage];
    setMessages(newMessages);

    // Reset inactivity timer after user sends message
    resetInactivityTimer();

    try {
      const newQAPairCount = qaPairCount + 1;

      console.log("[v0] Sending message with Q/A count:", {
        sessionId,
        currentQAPairCount: qaPairCount,
        newQAPairCount,
        conversationLoop,
      });

      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          patientId,
          email: email,
          query: currentInput,
          chatHistory: newMessages.map((msg) => ({
            role: msg.role,
            content: msg.content,
            timestamp:
              msg.timestamp instanceof Date
                ? msg.timestamp.toISOString()
                : msg.timestamp,
          })),
          sessionId: sessionId || undefined,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log("[v0] Received response:", data);

      if (!sessionId && data.sessionId) {
        setSessionId(data.sessionId);
        console.log("[v0] Session ID set:", data.sessionId);
      }

      if (data.shouldEndConversation) {
        console.log(
          "[v0] Backend detected end conversation, triggering summary...",
        );
        setMessages([
          ...newMessages,
          {
            role: "assistant",
            content: data.response.answer,
            timestamp: new Date(),
            agentType: "system",
          },
        ]);
        setLoading(false);
        await handleEndConversation();
        return;
      }

      let assistantMessage: Message | null = null;

      if (data.isCheckpoint) {
        console.log(
          "[v0] Backend triggered checkpoint at",
          newQAPairCount,
          "Q/A pairs",
        );
        // Update the Q/A count since this exchange is complete
        setQaPairCount(newQAPairCount);

        // Determine checkpoint type based on conversation loop
        if (conversationLoop === 1) {
          setCheckpointType("regular"); // First checkpoint at 6 Q&A
        } else {
          setCheckpointType("extended"); // Extended checkpoint at 3 Q&A
        }

        setLoading(false);
        // Show checkpoint dialog after brief delay
        setTimeout(() => {
          setShowCheckpoint(true);
        }, 500);
        return;
      }

      if (data.agentType === "emergency") {
        const content =
          data.response.message || data.response.answer || "Emergency response";
        assistantMessage = {
          role: "assistant",
          content,
          timestamp: new Date(),
          agentType: "emergency",
          emergencyNumber: data.response.emergencyNumber,
          nearbyClinicLocations: data.response.nearbyClinicLocations,
          needsLocation: data.response.needsLocation,
        };
      } else {
        const content =
          data.response?.answer ||
          data.response?.message ||
          "No response available";
        assistantMessage = {
          role: "assistant",
          content,
          timestamp: new Date(),
          agentType: data.agentType,
          severity: data.severity,
        };
      }

      if (assistantMessage) {
        const finalMessages = [...newMessages, assistantMessage];
        setMessages(finalMessages);

        if (data.agentType === "clinical") {
          setQaPairCount(newQAPairCount);

          console.log("[v0] Q/A pair count updated:", {
            previousCount: qaPairCount,
            newCount: newQAPairCount,
            conversationLoop,
          });
        }
      }

      setLoading(false);
      // Reset inactivity timer after receiving response
      resetInactivityTimer();
    } catch (error) {
      console.error("Chat error:", error);
      setError(
        error instanceof Error ? error.message : "Failed to send message",
      );
      setLoading(false);
      resetInactivityTimer();
    }
  };

  const displayMessages = messages;

  return (
    <Card className="w-full h-full max-w-2xl mx-auto flex flex-col">
      <CardHeader className="border-b">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-2xl">Harley Health Portal</CardTitle>
            <p className="text-sm text-gray-500 mt-2">
              Chat with our healthcare assistant
            </p>
            {sessionId && (
              <p className="text-xs text-gray-400 mt-1">
                Session: {sessionId.substring(0, 12)}... | Loop:{" "}
                {conversationLoop} | Q/A Pairs: {qaPairCount}/
                {conversationLoop === 1
                  ? INITIAL_CHECKPOINT
                  : EXTENDED_CHECKPOINT}
              </p>
            )}
          </div>
          <div className="flex gap-2">
            {messages.length > 0 && !showEndMessage && !conversationEnded && (
              <Button
                variant="outline"
                size="sm"
                onClick={handleEndConversation}
                className="flex items-center gap-2 bg-transparent hover:bg-red-50 hover:text-red-600 hover:border-red-200"
              >
                <LogOut className="w-4 h-4" />
                End Conversation
              </Button>
            )}
            {messages.length > 0 && (
              <Button
                variant="outline"
                size="sm"
                onClick={handleClearChat}
                className="flex items-center gap-2 bg-transparent"
              >
                <RefreshCw className="w-4 h-4" />
                Clear Chat
              </Button>
            )}
          </div>
        </div>
      </CardHeader>

      <CardContent className="flex-1 flex flex-col p-4 gap-4 overflow-auto">
        {showWelcome && (
          <div className="flex items-start gap-3 mb-4">
            <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center flex-shrink-0">
              <span className="text-blue-600 text-sm">🏥</span>
            </div>
            <div className="bg-gray-100 rounded-lg p-4 flex-1">
              <p className="text-sm font-semibold mb-2">Welcome to Harley</p>
              <p className="text-sm text-gray-700 mb-2">
                I'm your healthcare assistant. I can help with:
              </p>
              <ul className="text-sm text-gray-700 space-y-1">
                <li>• Clinical health questions</li>
                <li>• General health FAQs</li>
                <li>• Emergency situation assessment</li>
                <li>• Personal account information</li>
                <li>• Conversation history summaries</li>
              </ul>
              <p className="text-xs text-gray-500 mt-3">
                After {INITIAL_CHECKPOINT} Q/A exchanges, you'll be asked if you
                need more information. If you continue, after{" "}
                {EXTENDED_CHECKPOINT} more exchanges you'll be asked again.
              </p>
              <p className="text-sm text-yellow-800">
                After 40 seconds of inactivity, I'll check if you're there. If you don't respond,
                I will automatically end the session and send a summary to your doctor.
              </p>
            </div>
          </div>
        )}

        {/* Regular Checkpoint Prompt */}
        {showCheckpoint && checkpointType === "regular" && (
          <div className="bg-blue-50 border border-blue-300 rounded-lg p-4 flex flex-col gap-4">
            <div className="flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-sm font-semibold text-blue-900">
                  Conversation Checkpoint - {INITIAL_CHECKPOINT} Questions
                  Completed
                </p>
                <p className="text-sm text-blue-800 mt-2">
                  I have gathered information from {qaPairCount} questions.
                  Would you like to:
                </p>
                <ul className="text-sm text-blue-800 space-y-1 mt-2 ml-2">
                  <li>
                    • <strong>Continue:</strong> Provide additional information
                    ({EXTENDED_CHECKPOINT} more questions)
                  </li>
                  <li>
                    • <strong>End Session:</strong> Type <strong>"Yes"</strong> or <strong>"End Conversation"</strong> to create and send summary
                  </li>
                </ul>
                <p className="text-xs text-blue-700 mt-3 font-medium">
                  💬 You can explicitly confirm by typing <strong>"Yes"</strong> or <strong>"Ok"</strong>.
                </p>
              </div>
            </div>
            <div className="flex gap-3">
              <Button
                onClick={handleCheckpointContinue}
                className="flex-1 bg-blue-600 hover:bg-blue-700 text-white"
              >
                Continue ({EXTENDED_CHECKPOINT} more questions)
              </Button>
              <Button
                onClick={handleCheckpointEnd}
                variant="outline"
                className="flex-1 border-blue-300 text-blue-600 hover:bg-blue-100 bg-transparent"
              >
                End Session & Create Summary
              </Button>
            </div>
          </div>
        )}

        {/* Extended Checkpoint Prompt */}
        {showCheckpoint && checkpointType === "extended" && (
          <div className="bg-orange-50 border border-orange-300 rounded-lg p-4 flex flex-col gap-4">
            <div className="flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-orange-600 flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-sm font-semibold text-orange-900">
                  Extended Session Checkpoint
                </p>
                <p className="text-sm text-orange-800 mt-2">
                  You've provided {EXTENDED_CHECKPOINT} additional questions. To
                  ensure we have comprehensive information:
                </p>
                <ul className="text-sm text-orange-800 space-y-1 mt-2 ml-2">
                  <li>
                    • <strong>Continue:</strong> Provide more details (
                    {EXTENDED_CHECKPOINT} more questions)
                  </li>
                  <li>
                    • <strong>End Session:</strong> Type <strong>"End Conversation"</strong> to create summary with all
                    information gathered
                  </li>
                </ul>
                <p className="text-xs text-orange-700 mt-2">
                  Recommended: End session to allow your doctor to review the
                  comprehensive information already collected.
                </p>
                <p className="text-xs text-orange-700 mt-2 font-medium">
                  💬 Type <strong>"End Conversation"</strong> to conclude and send the summary.
                </p>
              </div>
            </div>
            <div className="flex gap-3">
              <Button
                onClick={handleCheckpointContinue}
                variant="outline"
                className="flex-1 border-orange-300 text-orange-600 hover:bg-orange-100 bg-transparent"
              >
                Continue ({EXTENDED_CHECKPOINT} more questions)
              </Button>
              <Button
                onClick={handleCheckpointEnd}
                className="flex-1 bg-orange-600 hover:bg-orange-700 text-white"
              >
                End Session & Create Summary
              </Button>
            </div>
          </div>
        )}

        {/* Inactivity Prompt */}
        {showInactivityPrompt && (
          <div className="bg-yellow-50 border border-yellow-300 rounded-lg p-4 flex flex-col gap-4">
            <div className="flex items-start gap-3">
              <Clock className="w-5 h-5 text-yellow-600 flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-sm font-semibold text-yellow-900">
                  Are You Still There?
                </p>
                <p className="text-sm text-yellow-800 mt-2">
                  I noticed you haven't sent a message for a while. Would you
                  like to:
                </p>
                <ul className="text-sm text-yellow-800 space-y-1 mt-2 ml-2">
                  <li>
                    • <strong>Continue:</strong> Keep chatting and providing
                    information
                  </li>
                  <li>
                    • <strong>End Session:</strong> Type <strong>"End Conversation"</strong> to create a summary and send it to your doctor
                  </li>
                </ul>
                <p className="text-xs text-yellow-700 mt-3 font-medium">
                  💬 Type <strong>"End Conversation"</strong> to conclude and send the summary.
                </p>
              </div>
            </div>
            <div className="flex gap-3">
              <Button
                onClick={() => handleInactivityResponse(false)}
                className="flex-1 bg-yellow-600 hover:bg-yellow-700 text-white"
              >
                Continue Chatting
              </Button>
              <Button
                onClick={() => handleInactivityResponse(true)}
                variant="outline"
                className="flex-1 border-yellow-300 text-yellow-600 hover:bg-yellow-100 bg-transparent"
              >
                End Session & Create Summary
              </Button>
            </div>
          </div>
        )}

        {showEndMessage && (
          <div className="bg-green-50 border border-green-300 rounded-lg p-4">
            <div className="flex items-start gap-3">
              <CheckCircle className="w-5 h-5 text-green-600 flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-sm font-semibold text-green-900">
                  Thank You!
                </p>
                <p className="text-sm text-green-800 mt-1">
                  Thanks for your information. I've created a summary and sent
                  it to your doctor. You can view it in the Communications tab.
                </p>
              </div>
            </div>
          </div>
        )}

        {showThankYou && (
          <div className="bg-green-50 border border-green-300 rounded-lg p-4">
            <div className="flex items-start gap-3">
              <CheckCircle className="w-5 h-5 text-green-600 flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-sm font-semibold text-green-900">
                  Thank You!
                </p>
                <p className="text-sm text-green-800 mt-1">
                  Thanks for your information. I've created a summary and sent
                  it to your doctor. You can view it in the Communications tab.
                </p>
              </div>
            </div>
          </div>
        )}

        {displayedSummary && (
          <div className="bg-white border border-gray-300 rounded-lg p-4">
            <h3 className="text-sm font-semibold text-gray-900 mb-2">
              Conversation Summary
            </h3>
            <div className="text-sm text-gray-700 whitespace-pre-wrap">
              {displayedSummary}
            </div>
          </div>
        )}

        {messages.map((message, index) => (
          <div
            key={index}
            className={`flex items-start gap-3 ${message.role === "user" ? "justify-end" : ""
              }`}
          >
            {message.role === "assistant" && (
              <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center flex-shrink-0">
                <span className="text-blue-600 text-sm">🏥</span>
              </div>
            )}
            <div
              className={`max-w-xs lg:max-w-md rounded-lg p-3 ${message.role === "user"
                ? "bg-blue-600 text-white"
                : "bg-gray-100 text-gray-900"
                }`}
            >
              <p className="text-sm whitespace-pre-wrap">{message.content}</p>
              <span
                className={`text-xs mt-1 block ${message.role === "user" ? "text-blue-100" : "text-gray-500"
                  }`}
              >
                {message.timestamp.toLocaleTimeString()}
              </span>
            </div>
            {message.role === "user" && (
              <div className="w-8 h-8 bg-gray-300 rounded-full flex items-center justify-center flex-shrink-0">
                <span className="text-gray-700 text-sm">👤</span>
              </div>
            )}
          </div>
        ))}

        {loading && (
          <div className="flex items-center gap-2 text-gray-500">
            <Loader2 className="w-4 h-4 animate-spin" />
            <span className="text-sm">Processing your query...</span>
          </div>
        )}

        {error && (
          <div className="flex items-center gap-2 text-red-500">
            <AlertCircle className="w-4 h-4" />
            <span className="text-sm">{error}</span>
          </div>
        )}

        <div ref={messagesEndRef} />
      </CardContent>

      <div className="border-t p-4 flex gap-2">
        <Input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={
            showCheckpoint || showInactivityPrompt
              ? "Type 'End Conversation' to end session or 'Continue' to chat..."
              : "Type your health question..."
          }
          onKeyPress={(e) => e.key === "Enter" && handleSendMessage()}
          disabled={loading || showEndMessage || conversationEnded}
          className="flex-1"
        />
        <Button
          onClick={handleSendMessage}
          disabled={
            loading || !input.trim() || showEndMessage || conversationEnded
          }
          className="bg-blue-600 hover:bg-blue-700"
        >
          {loading ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            <Send className="w-4 h-4" />
          )}
        </Button>
      </div>
    </Card>
  );
}
