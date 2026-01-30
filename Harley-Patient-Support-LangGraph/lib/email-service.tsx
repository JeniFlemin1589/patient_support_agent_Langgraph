import nodemailer from "nodemailer";

interface EmailPayload {
  to: string;
  subject: string;
  htmlContent: string;
  questions: Array<{ q: string; a: string }>;
}

export async function sendCommunicationEmail(
  payload: EmailPayload
): Promise<boolean> {
  // Note: Configure with your email provider (Gmail, SendGrid, etc.)
  const transporter = nodemailer.createTransport({
    service: "gmail",
    auth: {
      user: process.env.EMAIL_USER,
      pass: process.env.EMAIL_PASSWORD,
    },
  });

  const qaHtml = payload.questions
    .map(
      (item) =>
        `<div style="margin: 15px 0; padding: 10px; border-left: 3px solid #4CAF50;">
        <p><strong>Q: ${item.q}</strong></p>
        <p>A: ${item.a}</p>
      </div>`
    )
    .join("");

  const html = `
    <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
      <h2 style="color: #333;">Your Healthcare Portal Communication</h2>
      <p>Below is a summary of your recent interaction with our healthcare portal:</p>
      ${qaHtml}
      <p style="margin-top: 20px; color: #666; font-size: 12px;">
        This is an automated email from Harley Health Portal. Please do not reply to this email.
        Log in to your portal to view complete details.
      </p>
    </div>
  `;

  try {
    await transporter.sendMail({
      from: process.env.EMAIL_USER,
      to: payload.to,
      subject: payload.subject,
      html,
    });
    return true;
  } catch (error) {
    console.error("Email send failed:", error);
    return false;
  }
}

export async function sendEmergencyAlert(
  patientEmail: string,
  emergencyDetails: string
): Promise<boolean> {
  const transporter = nodemailer.createTransport({
    service: "gmail",
    auth: {
      user: process.env.EMAIL_USER,
      pass: process.env.EMAIL_PASSWORD,
    },
  });

  const html = `
    <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; background: #fff3cd; border: 2px solid #ff6b6b; padding: 20px;">
      <h2 style="color: #d32f2f;">EMERGENCY ALERT - IMMEDIATE ACTION REQUIRED</h2>
      <p>${emergencyDetails}</p>
      <p style="color: #d32f2f;"><strong>Please contact emergency services or visit the nearest emergency room immediately.</strong></p>
    </div>
  `;

  try {
    await transporter.sendMail({
      from: process.env.EMAIL_USER,
      to: patientEmail,
      subject: "URGENT: Emergency Alert from Harley Health",
      html,
    });
    return true;
  } catch (error) {
    console.error("Emergency email send failed:", error);
    return false;
  }
}

export async function sendEmergencySummaryToDoctor(
  doctorEmail: string,
  patientName: string,
  patientContact: string,
  emergencyDescription: string,
  conversationSummary: string
): Promise<boolean> {
  const transporter = nodemailer.createTransport({
    service: "gmail",
    auth: {
      user: process.env.EMAIL_USER,
      pass: process.env.EMAIL_PASSWORD,
    },
  });

  const html = `
    <div style="font-family: Arial, sans-serif; max-width: 700px; margin: 0 auto; background: #fee; border: 3px solid #f00; padding: 20px;">
      <h2 style="color: #d32f2f; text-align: center;">🚨 EMERGENCY ALERT - PATIENT REQUIRES IMMEDIATE ATTENTION</h2>
      
      <div style="background: #fff; padding: 15px; border-radius: 5px; margin: 15px 0;">
        <h3 style="color: #333; margin-top: 0;">Patient Information:</h3>
        <p><strong>Name:</strong> ${patientName}</p>
        <p><strong>Contact:</strong> ${patientContact}</p>
        <p><strong>Report Time:</strong> ${new Date().toLocaleString()}</p>
      </div>

      <div style="background: #ffe; padding: 15px; border-radius: 5px; margin: 15px 0; border-left: 4px solid #d32f2f;">
        <h3 style="color: #d32f2f; margin-top: 0;">Emergency Description:</h3>
        <p>${emergencyDescription}</p>
      </div>

      <div style="background: #f5f5f5; padding: 15px; border-radius: 5px; margin: 15px 0;">
        <h3 style="color: #333; margin-top: 0;">Conversation Summary:</h3>
        <p style="white-space: pre-wrap; line-height: 1.6;">${conversationSummary}</p>
      </div>

      <div style="background: #fee; padding: 15px; border-radius: 5px; text-align: center; margin: 15px 0;">
        <p style="color: #d32f2f; font-size: 14px;"><strong>Action Required:</strong> Please review this case immediately and contact the patient if necessary.</p>
      </div>

      <p style="margin-top: 20px; color: #666; font-size: 12px; text-align: center;">
        This is an automated emergency alert from Harley Health Portal. Emergency summaries are only shared with authorized medical professionals.
      </p>
    </div>
  `;

  try {
    await transporter.sendMail({
      from: process.env.EMAIL_USER,
      to: doctorEmail,
      subject: `🚨 URGENT - Emergency Alert: Patient ${patientName} - Immediate Action Required`,
      html,
    });
    console.log("[v0] Emergency summary sent to doctor:", doctorEmail);
    return true;
  } catch (error) {
    console.error("[v0] Failed to send emergency summary to doctor:", error);
    return false;
  }
}
