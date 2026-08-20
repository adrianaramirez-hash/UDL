import { SIACard } from "@/components/cards/SIACard"

type QuestionCardProps = {
  title: string
  description: string
}

export function QuestionCard({
  title,
  description,
}: QuestionCardProps) {
  return (
    <SIACard
      interactive
      className="cursor-pointer p-6"
    >
      <h3 className="text-xl font-semibold">
        {title}
      </h3>

      <p className="mt-3 text-muted-foreground">
        {description}
      </p>

      <p className="mt-6 text-sm font-medium text-sky-700">
        Ver análisis →
      </p>
    </SIACard>
  )
}