import { SIACard } from "@/components/cards/SIACard"
import { Link } from "react-router-dom"

type QuestionCardProps = {
  to: string
  title: string
  description: string
}

export function QuestionCard({
  to,
  title,
  description,
}: QuestionCardProps) {
  return (
    <Link to={to} className="block rounded-2xl focus-visible:outline-2 focus-visible:outline-offset-4">
      <SIACard interactive className="flex h-full flex-col p-6">
        <h3 className="text-xl font-semibold">{title}</h3>
        <p className="mt-3 flex-1 text-muted-foreground">{description}</p>
        <p className="mt-6 text-sm font-medium text-sky-700">Ver análisis →</p>
      </SIACard>
    </Link>
  )
}
