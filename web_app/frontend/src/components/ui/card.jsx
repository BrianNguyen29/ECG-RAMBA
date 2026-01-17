import * as React from "react"
import { cn } from "../../lib/utils"
import { motion } from "framer-motion"

// Base Card
const Card = React.forwardRef(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn(
      "rounded-2xl border border-slate-800 bg-slate-900 text-slate-100 shadow-xl shadow-black/20",
      className
    )}
    {...props}
  />
))
Card.displayName = "Card"

// Glass Card (Premium Medical Variant)
const GlassCard = React.forwardRef(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn(
      "rounded-2xl border border-white/10 bg-slate-900/70 backdrop-blur-xl text-slate-100 shadow-xl shadow-black/30",
      className
    )}
    {...props}
  />
))
GlassCard.displayName = "GlassCard"

// Animated Card with Framer Motion
const AnimatedCard = React.forwardRef(({ className, delay = 0, ...props }, ref) => (
  <motion.div
    ref={ref}
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ duration: 0.4, delay, ease: "easeOut" }}
    whileHover={{ scale: 1.01, boxShadow: "0 20px 40px -10px rgba(0,0,0,0.3)" }}
    className={cn(
      "rounded-2xl border border-slate-800 bg-slate-900 text-slate-100 shadow-xl shadow-black/20 transition-all",
      className
    )}
    {...props}
  />
))
AnimatedCard.displayName = "AnimatedCard"

// Metric Card (KPI Display)
const MetricCard = React.forwardRef(({ 
  title, 
  value, 
  subtitle, 
  trend, 
  trendValue,
  icon: Icon,
  color = "blue",
  className, 
  ...props 
}, ref) => {
  const colorMap = {
    blue: "from-blue-500 to-indigo-600",
    green: "from-emerald-500 to-teal-600",
    red: "from-rose-500 to-red-600",
    purple: "from-violet-500 to-purple-600",
    amber: "from-amber-500 to-orange-600"
  }
  
  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      whileHover={{ scale: 1.02 }}
      transition={{ duration: 0.3 }}
      className={cn(
        "relative overflow-hidden rounded-2xl bg-slate-900 border border-slate-800 p-6 shadow-lg",
        className
      )}
      {...props}
    >
      {/* Gradient Accent */}
      <div className={cn("absolute top-0 left-0 right-0 h-1 bg-gradient-to-r", colorMap[color])} />
      
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs font-bold text-slate-500 uppercase tracking-wider">{title}</p>
          <p className="mt-2 text-2xl font-bold text-slate-100">{value}</p>
          {subtitle && <p className="mt-1 text-xs text-slate-400">{subtitle}</p>}
        </div>
        {Icon && (
          <div className={cn("p-2 rounded-lg bg-gradient-to-br text-white shadow-lg", colorMap[color])}>
            <Icon className="w-5 h-5" />
          </div>
        )}
      </div>
      
      {trend && (
        <div className="mt-4 flex items-center gap-2">
          <span className={cn(
            "text-sm font-semibold",
            trend === "up" ? "text-emerald-500" : "text-rose-500"
          )}>
            {trend === "up" ? "↑" : "↓"} {trendValue}
          </span>
          <span className="text-xs text-slate-500">vs last period</span>
        </div>
      )}
    </motion.div>
  )
})
MetricCard.displayName = "MetricCard"

const CardHeader = React.forwardRef(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn("flex flex-col space-y-1.5 p-6 border-b border-slate-800", className)}
    {...props}
  />
))
CardHeader.displayName = "CardHeader"

const CardTitle = React.forwardRef(({ className, ...props }, ref) => (
  <h3
    ref={ref}
    className={cn(
      "text-lg font-semibold leading-none tracking-tight text-slate-100 flex items-center gap-2",
      className
    )}
    {...props}
  />
))
CardTitle.displayName = "CardTitle"

const CardDescription = React.forwardRef(({ className, ...props }, ref) => (
  <p
    ref={ref}
    className={cn("text-sm text-slate-400", className)}
    {...props}
  />
))
CardDescription.displayName = "CardDescription"

const CardContent = React.forwardRef(({ className, ...props }, ref) => (
  <div ref={ref} className={cn("p-6 pt-0 mt-4", className)} {...props} />
))
CardContent.displayName = "CardContent"

const CardFooter = React.forwardRef(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn("flex items-center p-6 pt-0 border-t border-slate-800 mt-4", className)}
    {...props}
  />
))
CardFooter.displayName = "CardFooter"

export { 
  Card, 
  GlassCard,
  AnimatedCard,
  MetricCard,
  CardHeader, 
  CardFooter, 
  CardTitle, 
  CardDescription, 
  CardContent 
}
