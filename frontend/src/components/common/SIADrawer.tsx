import { useEffect } from "react"
import type { ReactNode } from "react"

type SIADrawerProps = {
  open: boolean
  title?: string
  width?: "md" | "lg" | "xl"
  onClose: () => void
  children: ReactNode
}

const widths = {
  md: "max-w-xl",
  lg: "max-w-2xl",
  xl: "max-w-4xl",
}

export function SIADrawer({
  open,
  title,
  width = "lg",
  onClose,
  children,
}: SIADrawerProps) {
  useEffect(() => {
    function handleEscape(event: KeyboardEvent) {
      if (event.key === "Escape") {
        onClose()
      }
    }

    if (open) {
      document.body.style.overflow = "hidden"
      window.addEventListener("keydown", handleEscape)
    }

    return () => {
      document.body.style.overflow = ""
      window.removeEventListener(
        "keydown",
        handleEscape,
      )
    }
  }, [open, onClose])

  if (!open) {
    return null
  }

  return (
    <div className="fixed inset-0 z-50">
      <div
        className="absolute inset-0 bg-black/40 backdrop-blur-[2px]"
        onClick={onClose}
      />

      <aside
        className={`absolute right-0 top-0 h-full w-full ${widths[width]} bg-background shadow-2xl`}
      >
        <div className="flex h-full flex-col">

          <div className="flex items-center justify-between border-b px-6 py-5">

            <div>

              {title && (
                <h2 className="text-xl font-semibold">
                  {title}
                </h2>
              )}

            </div>

            <button
              onClick={onClose}
              className="rounded-lg p-2 transition hover:bg-muted"
            >
              ✕
            </button>

          </div>

          <div className="flex-1 overflow-y-auto p-6">
            {children}
          </div>

        </div>
      </aside>
    </div>
  )
}

