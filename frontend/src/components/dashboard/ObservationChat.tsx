import { FormEvent, useState } from "react"

import { SIACard } from "@/components/cards/SIACard"
import {
  askObservationAdvisor,
  type ObservationChatContext,
  type ObservationChatResponse,
} from "@/services/observationChatService"

type ChatMessage = {
  id: number
  role: "user" | "assistant"
  text: string
}

const INITIAL_MESSAGES: ChatMessage[] = [
  {
    id: 1,
    role: "assistant",
    text:
      "Soy el Asesor SIA de Observación de Clases. Puedes preguntarme por docentes, carreras, materias, promedios, observaciones y casos prioritarios.",
  },
]

export function ObservationChat() {
  const [messages, setMessages] =
    useState<ChatMessage[]>(INITIAL_MESSAGES)

  const [question, setQuestion] = useState("")

  const [context, setContext] =
    useState<ObservationChatContext>({})

  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function handleSubmit(
    event: FormEvent<HTMLFormElement>,
  ) {
    event.preventDefault()

    const trimmedQuestion = question.trim()

    if (!trimmedQuestion || loading) {
      return
    }

    const userMessage: ChatMessage = {
      id: Date.now(),
      role: "user",
      text: trimmedQuestion,
    }

    setMessages((current) => [
      ...current,
      userMessage,
    ])

    setQuestion("")
    setLoading(true)
    setError(null)

    try {
      const response: ObservationChatResponse =
        await askObservationAdvisor(
          trimmedQuestion,
          context,
        )

      const assistantMessage: ChatMessage = {
        id: Date.now() + 1,
        role: "assistant",
        text: response.respuesta,
      }

      setMessages((current) => [
        ...current,
        assistantMessage,
      ])

      setContext(
        response.contexto ?? {},
      )
    } catch (err) {
      console.error(err)

      setError(
        "No fue posible consultar al Asesor SIA.",
      )
    } finally {
      setLoading(false)
    }
  }

  function clearConversation() {
    setMessages(INITIAL_MESSAGES)
    setContext({})
    setQuestion("")
    setError(null)
  }

  return (
    <SIACard className="overflow-hidden">
      <div className="border-b px-6 py-5">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <p className="sia-eyebrow">
              ASESOR SIA
            </p>

            <h2 className="mt-2 text-2xl sia-heading">
              Pregunta sobre Observación de Clases
            </h2>

            <p className="mt-2 text-sm text-muted-foreground">
              Consulta docentes, carreras, materias, promedios,
              observaciones y seguimiento.
            </p>
          </div>

          <button
            type="button"
            onClick={clearConversation}
            className="shrink-0 rounded-xl border bg-white px-3 py-2 text-xs font-medium text-slate-600 transition hover:bg-slate-50"
          >
            Nueva conversación
          </button>
        </div>

        {context.ultimo_docente && (
          <div className="mt-4 inline-flex items-center gap-2 rounded-full border border-primary/10 bg-primary/5 px-3 py-1.5 text-xs">
            <span className="text-slate-500">
              Contexto:
            </span>

            <span className="font-medium text-primary">
              {context.ultimo_docente}
            </span>
          </div>
        )}
      </div>

      <div className="flex min-h-[420px] flex-col">
        <div className="flex-1 space-y-4 overflow-y-auto bg-slate-50/40 p-6">
          {messages.map((message) => {
            const isUser =
              message.role === "user"

            return (
              <div
                key={message.id}
                className={`flex ${
                  isUser
                    ? "justify-end"
                    : "justify-start"
                }`}
              >
                <div
                  className={`max-w-[85%] rounded-2xl px-4 py-3 text-sm leading-6 shadow-sm ${
                    isUser
                      ? "bg-primary text-white"
                      : "border bg-white text-slate-700"
                  }`}
                >
                  {message.text}
                </div>
              </div>
            )
          })}

          {loading && (
            <div className="flex justify-start">
              <div className="rounded-2xl border bg-white px-4 py-3 text-sm text-muted-foreground shadow-sm">
                Analizando información...
              </div>
            </div>
          )}

          {error && (
            <div className="rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">
              {error}
            </div>
          )}
        </div>

        <form
          onSubmit={handleSubmit}
          className="border-t bg-white p-4"
        >
          <div className="flex gap-3">
            <input
              type="text"
              value={question}
              onChange={(event) =>
                setQuestion(event.target.value)
              }
              placeholder="Ej. Busca a Diego Molina"
              className="min-w-0 flex-1 rounded-xl border bg-white px-4 py-3 text-sm outline-none transition focus:border-primary/40 focus:ring-2 focus:ring-primary/10"
            />

            <button
              type="submit"
              disabled={
                loading || !question.trim()
              }
              className="rounded-xl bg-primary px-5 py-3 text-sm font-medium text-white transition hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
            >
              Enviar
            </button>
          </div>

          <div className="mt-3 flex flex-wrap gap-2">
            {[
              "Busca a Diego Molina",
              "¿Qué fortalezas tiene?",
              "¿Qué recomendaciones recibió?",
              "¿En qué materia fue observado?",
              "¿En qué corte?",
            ].map((suggestion) => (
              <button
                key={suggestion}
                type="button"
                onClick={() =>
                  setQuestion(suggestion)
                }
                className="rounded-full border bg-white px-3 py-1.5 text-xs text-slate-600 transition hover:bg-slate-50"
              >
                {suggestion}
              </button>
            ))}
          </div>
        </form>
      </div>
    </SIACard>
  )
}