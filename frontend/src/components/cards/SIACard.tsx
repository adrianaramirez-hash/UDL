import type { ReactNode } from "react"

type SIACardProps = {
  children: ReactNode
  className?: string
  interactive?: boolean
}

export function SIACard({
  children,
  className = "",
  interactive = false,
}: SIACardProps) {
  return (
    <div
      className={[
        "sia-card",
        interactive ? "sia-card-interactive" : "",
        className,
      ]
        .filter(Boolean)
        .join(" ")}
    >
      {children}
    </div>
  )
}