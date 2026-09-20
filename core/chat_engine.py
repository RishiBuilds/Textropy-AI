import os
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

CHAT_MODEL = "meta-llama/llama-4-maverick:free"

MAX_CONTEXT_CHARS = int(os.getenv("CHAT_CONTEXT_CHARS", "24000"))

async def chat_with_document(extracted_text, user_question, chat_history=None):
    if not os.getenv('OPENROUTER_API_KEY'):
        raise ValueError("Please provide an OpenRouter API Key in your .env file.")

    client = AsyncOpenAI(
        api_key=os.getenv('OPENROUTER_API_KEY'),
        base_url="https://openrouter.ai/api/v1",
    )

    system_message = f"""You are a helpful document assistant for Textropy AI. The user has extracted text from a document using OCR. Answer their questions about the document accurately and concisely.

DOCUMENT CONTENT:
---
{extracted_text[:MAX_CONTEXT_CHARS]}
---

RULES:
1. Answer based ONLY on the document content above.
2. If the user asks to summarize, provide a clear and concise summary.
3. If the user asks about an equation, explain it step by step using proper LaTeX formatting with $$ delimiters.
4. If the user asks to translate, translate the document content to the requested language.
5. If you cannot answer from the document, say so clearly.
6. Use markdown formatting for readability.
7. Keep LaTeX math wrapped in $$ for block equations or $ for inline."""

    messages = [{"role": "system", "content": system_message}]

    if chat_history:
        for msg in chat_history:
            messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": user_question})

    try:
        completion = await client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            max_tokens=2000,
            temperature=0.3
        )
        content = completion.choices[0].message.content
        if content is None:
            return "The model returned an empty response. Please try again."
        return content
    except Exception as e:
        raise Exception(f"Chat failed: {str(e)}")

QUICK_ACTIONS = {
    "summarize": "Summarize this document in a clear, concise format with key points.",
    "explain_equations": "Identify and explain all mathematical equations in this document step by step.",
    "translate_english": "Translate the entire document content to English.",
    "translate_hindi": "Translate the entire document content to Hindi.",
    "key_formulas": "List all key formulas and equations found in this document with brief explanations.",
    "simplify": "Rewrite this document in simpler language that a high school student could understand."
}
