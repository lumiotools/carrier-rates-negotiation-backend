import {
  ContextChatEngine,
  OpenAI,
  VectorStoreIndex,
  storageContextFromDefaults,
} from "llamaindex";
import dotenv from "dotenv";
import fs from "fs";
import path from "path";

import express from "express";
import cors from "cors";

dotenv.config();

const app = express();
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(
  cors({
    origin: "*",
  })
);

const PORT = process.env.PORT || 8000;

const getStorageContext = async (carrier_url) => {
  for (let file of fs.readdirSync("negotiation-data")) {
    if (file === carrier_url) {
      const storageContext = await storageContextFromDefaults({
        persistDir: path.join("negotiation-data", carrier_url),
      });

      return storageContext;
    }
  }

  return null;
};

app.get("/api/carriers", async (req, res) => {
  const carrier_urls = fs.readdirSync("negotiation-data");

  const carriers = [];

  carrier_urls.forEach((carrier_url) => {
    let carrier_name = carrier_url.split(".")[0];
    carrier_name =
      carrier_name.substring(0, 1).toUpperCase() + carrier_name.substring(1);

    carriers.push({
      label: carrier_name,
      value: carrier_url,
    });
  });
  return res.status(200).json({ carriers });
});

app.post("/api/rates-negotiation-chat", async (req, res) => {
  const { carrier_url, chat_history, message } = req.body;

  if (!carrier_url || !chat_history || !message) {
    return res.status(400).send("Invalid request");
  }

  const storageContext = await getStorageContext(carrier_url);
  if (!storageContext) {
    return res.status(404).send("Carrier not found");
  }

  const index = await VectorStoreIndex.init({
    logProgress: true,
    storageContext: storageContext,
  });

  const retriever = index.asRetriever();

  const systemMessage = {
    role: "system",
    content: `
    You are a shipping assistant AI with expertise in rates negotiation for ${carrier_url}. Your primary goal is to provide accurate, contextually relevant, and user-focused answers to queries related to rates negotiation with ${carrier_url}. You have access to all the relevant data and insights required to evaluate and recommend the best rates negotiation strategies, pricing models, and cost-saving opportunities for ${carrier_url}.

    When responding to user queries:

    ### 1. **Core Information to Include for Rates Negotiation:**
        - **Negotiation Strategies**: Provide insights on effective negotiation strategies and tactics.
        - **Pricing Models**: Explain different pricing models and their benefits for ${carrier_url}.
        - **Cost-Saving Opportunities**: Identify cost-saving opportunities and recommend ways to optimize rates.

    ### 2. **Handling Queries:**
    - **Negotiation Strategies**: Offer negotiation strategies tailored to the user's requirements.
    - **Pricing Models**: Explain the pricing models available for ${carrier_url} and their advantages.
    - **Cost-Saving Opportunities**: Suggest ways to reduce costs and optimize rates for ${carrier_url}.
    - **Evaluation Criteria**: Use the data provided to evaluate and recommend the best rates negotiation practices.

    ### 3. **Clarity and Detail:**
    - **Terminology**: Use clear, user-friendly language and avoid jargon.
    - **Explanations**: Provide detailed explanations to enhance the user's understanding.
  
    ### 4. **Resources:**
    - If the user requests links or additional resources, provide accurate and reliable links to support their inquiry.
    - Avoid broken, incomplete, or outdated links.    
    `,
  };

  const chatEngine = new ContextChatEngine({
    retriever: retriever,
    systemPrompt: systemMessage.content,
    contextRole: "system",
    chatModel: new OpenAI({
      apiKey: process.env.OPENAI_API_KEY,
      model: "gpt-4o-mini",
    }),
  });

  console.log([systemMessage, ...chat_history]);

  const response = await chatEngine.chat({
    chatHistory: [systemMessage, ...chat_history],
    message: message,
  });

  return res.status(200).json({ response: response.message.content });
});

app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
