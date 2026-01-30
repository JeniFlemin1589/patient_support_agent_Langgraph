// app/api/chat-history/summaries/[id]/route.ts
import { type NextRequest, NextResponse } from "next/server";
import { getCollection } from "@/lib/mongodb";
import { ObjectId } from "mongodb";

/**
 * GET endpoint to retrieve a specific summary by ID
 */
export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  // ← params is now a Promise
  try {
    const { id } = await params; // ← await params here

    // Validate ObjectId
    if (!ObjectId.isValid(id)) {
      return NextResponse.json(
        { error: "Invalid summary ID format" },
        { status: 400 }
      );
    }

    const chatHistoryCollection = await getCollection("chat_history");

    const summary = await chatHistoryCollection.findOne({
      _id: new ObjectId(id),
      isConversationSummary: true,
    });

    if (!summary) {
      return NextResponse.json({ error: "Summary not found" }, { status: 404 });
    }

    console.log(`[API] Retrieved summary ${id} from chat_history collection`);

    return NextResponse.json({
      success: true,
      summary: {
        _id: summary._id?.toString() || "",
        patientId: summary.patientId,
        patientEmail: summary.patientEmail,
        type: summary.communicationType || "clinical",
        summary: summary.summary || "",
        status: summary.status || "completed",
        severity: summary.severity || "medium",
        createdAt: summary.createdAt,
        timestamp: summary.createdAt,
        messageCount: summary.messageCount || 0,
        sessionId: summary.sessionId,
        qaPairCount: summary.qaPairCount,
        sentToDoctor: summary.sentToDoctor !== false,
        sentToPatient: summary.sentToPatient !== false,
      },
    });
  } catch (error) {
    console.error("[API] Error fetching summary:", error);
    return NextResponse.json(
      { error: "Failed to fetch summary" },
      { status: 500 }
    );
  }
}

/**
 * PATCH endpoint to update a specific summary by ID
 */
export async function PATCH(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  // ← params is now a Promise
  try {
    const { id } = await params; // ← await params here
    console.log(id);

    // Validate ObjectId
    if (!ObjectId.isValid(id)) {
      return NextResponse.json(
        { error: "Invalid summary ID format" },
        { status: 400 }
      );
    }

    const body = await request.json();
    const { summary, severity, status, notes } = body;

    // Validate at least one field is provided for update
    if (!summary && !severity && !status && !notes) {
      return NextResponse.json(
        { error: "At least one field must be provided for update" },
        { status: 400 }
      );
    }

    // Build update object
    const updateFields: any = {
      updatedAt: new Date(),
    };

    if (summary !== undefined) {
      updateFields.summary = summary;
    }

    if (severity !== undefined) {
      // Validate severity
      const validSeverities = ["low", "medium", "high", "critical"];
      if (!validSeverities.includes(severity)) {
        return NextResponse.json(
          {
            error: "Invalid severity. Must be: low, medium, high, or critical",
          },
          { status: 400 }
        );
      }
      updateFields.severity = severity;
    }

    if (status !== undefined) {
      // Validate status
      const validStatuses = ["completed", "in_progress", "pending", "archived"];
      if (!validStatuses.includes(status)) {
        return NextResponse.json(
          {
            error:
              "Invalid status. Must be: completed, in_progress, pending, or archived",
          },
          { status: 400 }
        );
      }
      updateFields.status = status;
    }

    if (notes !== undefined) {
      updateFields.notes = notes;
    }

    const chatHistoryCollection = await getCollection("chat_history");

    // Update the summary
    const result = await chatHistoryCollection.updateOne(
      {
        _id: new ObjectId(id),
        isConversationSummary: true,
      },
      {
        $set: updateFields,
      }
    );

    if (result.matchedCount === 0) {
      return NextResponse.json({ error: "Summary not found" }, { status: 404 });
    }

    console.log(
      `[API] Updated summary ${id} in chat_history collection`,
      updateFields
    );

    // Fetch and return the updated summary
    const updatedSummary = await chatHistoryCollection.findOne({
      _id: new ObjectId(id),
    });

    return NextResponse.json({
      success: true,
      message: "Summary updated successfully",
      summary: {
        _id: updatedSummary?._id?.toString() || "",
        patientId: updatedSummary?.patientId,
        patientEmail: updatedSummary?.patientEmail,
        type: updatedSummary?.communicationType || "clinical",
        summary: updatedSummary?.summary || "",
        status: updatedSummary?.status || "completed",
        severity: updatedSummary?.severity || "medium",
        createdAt: updatedSummary?.createdAt,
        updatedAt: updatedSummary?.updatedAt,
        timestamp: updatedSummary?.createdAt,
        messageCount: updatedSummary?.messageCount || 0,
        sessionId: updatedSummary?.sessionId,
        qaPairCount: updatedSummary?.qaPairCount,
        notes: updatedSummary?.notes,
        sentToDoctor: updatedSummary?.sentToDoctor !== false,
        sentToPatient: updatedSummary?.sentToPatient !== false,
      },
    });
  } catch (error) {
    console.error("[API] Error updating summary:", error);
    return NextResponse.json(
      {
        error: "Failed to update summary",
        details: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 }
    );
  }
}

/**
 * DELETE endpoint to delete a specific summary by ID
 */
export async function DELETE(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  // ← params is now a Promise
  try {
    const { id } = await params; // ← await params here

    // Validate ObjectId
    if (!ObjectId.isValid(id)) {
      return NextResponse.json(
        { error: "Invalid summary ID format" },
        { status: 400 }
      );
    }

    const chatHistoryCollection = await getCollection("chat_history");

    // Delete the summary
    const result = await chatHistoryCollection.deleteOne({
      _id: new ObjectId(id),
      isConversationSummary: true,
    });

    if (result.deletedCount === 0) {
      return NextResponse.json({ error: "Summary not found" }, { status: 404 });
    }

    console.log(`[API] Deleted summary ${id} from chat_history collection`);

    return NextResponse.json({
      success: true,
      message: "Summary deleted successfully",
      deletedId: id,
    });
  } catch (error) {
    console.error("[API] Error deleting summary:", error);
    return NextResponse.json(
      {
        error: "Failed to delete summary",
        details: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 }
    );
  }
}
