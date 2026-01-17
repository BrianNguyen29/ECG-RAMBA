import { useToast } from "./use-toast"
import { X } from "lucide-react"

export function Toaster() {
  const { toasts, dismiss } = useToast()

  return (
    <div className="fixed top-0 right-0 z-[100] flex flex-col gap-2 p-4 w-full max-w-[420px]">
      {toasts.map(function ({ id, title, description, action, ...props }) {
        return (
          <div 
            key={id} 
            className="group pointer-events-auto relative flex w-full items-center justify-between space-x-4 overflow-hidden rounded-md border border-slate-200 bg-white p-6 pr-8 shadow-lg transition-all"
            {...props}
          >
            <div className="grid gap-1">
              {title && <h3 className="text-sm font-semibold">{title}</h3>}
              {description && (
                <div className="text-sm opacity-90">{description}</div>
              )}
            </div>
            {action}
            <button 
                onClick={() => dismiss(id)}
                className="absolute right-2 top-2 rounded-md p-1 text-slate-500 opacity-0 transition-opacity hover:text-slate-900 focus:opacity-100 focus:outline-none focus:ring-2 group-hover:opacity-100"
            >
                <X className="h-4 w-4" />
            </button>
          </div>
        )
      })}
    </div>
  )
}
