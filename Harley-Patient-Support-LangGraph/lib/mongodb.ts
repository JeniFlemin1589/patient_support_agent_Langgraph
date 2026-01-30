import { MongoClient, type Db, type Collection, type Document } from "mongodb";

const MONGO_URI = process.env.NEXT_PUBLIC_MONGO_CONNECTION_STRING!;
const MONGO_USERNAME = process.env.NEXT_PUBLIC_MONGO_USERNAME!;
const MONGO_PASSWORD = process.env.NEXT_PUBLIC_MONGO_PASSWORD!;

let cachedClient: MongoClient | null = null;
let cachedDb: Db | null = null;

export async function connectToDatabase() {
  if (cachedClient && cachedDb) {
    return { client: cachedClient, db: cachedDb };
  }

  const connectionString = MONGO_URI.includes("@")
    ? MONGO_URI
    : `mongodb+srv://${MONGO_USERNAME}:${MONGO_PASSWORD}@${
        MONGO_URI.split("mongodb+srv://")[1]
      }`;

  const client = new MongoClient(connectionString);

  await client.connect();
  const db = client.db("harley_health");

  cachedClient = client;
  cachedDb = db;

  return { client, db };
}

export async function getCollection<T extends Document>(
  collectionName: string
): Promise<Collection<T>> {
  const { db } = await connectToDatabase();
  return db.collection<T>(collectionName);
}

// Initialize collections
export async function initializeCollections() {
  const { db } = await connectToDatabase();

  // Patients collection
  await db.collection("patients").createIndex({ email: 1 }, { unique: true });

  // Chat history collection
  await db.collection("chat_history").createIndex({ patientId: 1 });
  await db.collection("chat_history").createIndex({ createdAt: -1 });
  await db.collection("chat_history").createIndex({ sessionId: 1 });

  // Communications collection
  await db.collection("communications").createIndex({ patientId: 1 });
  await db.collection("communications").createIndex({ type: 1 });

  // Clinical notes collection
  await db.collection("clinical_notes").createIndex({ patientId: 1 });
}
