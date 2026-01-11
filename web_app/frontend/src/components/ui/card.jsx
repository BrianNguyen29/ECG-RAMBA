import * as React from "react"
import { cn } from "../../lib/utils"
import { motion } from "framer-motion"

// Base Card
const Card = React.forwardRef(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn(
      "rounded-2xl border border-gray-100 bg-white text-gray-950 shadow-lg shadow-gray-200/50",
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
      "rounded-2xl border border-white/20 bg-white/70 backdrop-blur-xl text-gray-950 shadow-xl shadow-gray-200/30",
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
    whileHover={{ scale: 1.01, boxShadow: "0 20px 40px -10px rgba(0,0,0,0.1)" }}
    className={cn(
      "rounded-2xl border border-gray-100 bg-white text-gray-950 shadow-lg shadow-gray-200/50 transition-all",
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
        "relative overflow-hidden rounded-2xl bg-white border border-gray-100 p-6 shadow-lg",
        className
      )}
      {...props}
    >
      {/* Gradient Accent */}
      <div className={cn("absolute top-0 left-0 right-0 h-1 bg-gradient-to-r", colorMap[color])} />
      
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm font-medium text-gray-500 uppercase tracking-wider">{title}</p>
          <p className="mt-2 text-3xl font-bold text-gray-900">{value}</p>
          {subtitle && <p className="mt-1 text-sm text-gray-400">{subtitle}</p>}
        </div>
        {Icon && (
          <div className={cn("p-3 rounded-xl bg-gradient-to-br text-white", colorMap[color])}>
            <Icon className="w-6 h-6" />
          </div>
        )}
      </div>
      
      {trend && (
        <div className="mt-4 flex items-center gap-2">
          <span className={cn(
            "text-sm font-semibold",
            trend === "up" ? "text-emerald-600" : "text-rose-600"
          )}>
            {trend === "up" ? "↑" : "↓"} {trendValue}
          </span>
          <span className="text-xs text-gray-400">vs last period</span>
        </div>
      )}
    </motion.div>
  )
})
MetricCard.displayName = "MetricCard"

const CardHeader = React.forwardRef(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn("flex flex-col space-y-1.5 p-6 border-b border-gray-100/50", className)}
    {...props}
  />
))
CardHeader.displayName = "CardHeader"

const CardTitle = React.forwardRef(({ className, ...props }, ref) => (
  <h3
    ref={ref}
    className={cn(
      "text-lg font-semibold leading-none tracking-tight text-gray-900 flex items-center gap-2",
      className
    )}
    {...props}
  />
))
CardTitle.displayName = "CardTitle"

const CardDescription = React.forwardRef(({ className, ...props }, ref) => (
  <p
    ref={ref}
    className={cn("text-sm text-gray-500", className)}
    {...props}
  />
))
CardDescription.displayName = "CardDescription"

const CardContent = React.forwardRef(({ className, ...props }, ref) => (
  <div ref={ref} className={cn("p-6 pt-0", className)} {...props} />
))
CardContent.displayName = "CardContent"

const CardFooter = React.forwardRef(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn("flex items-center p-6 pt-0", className)}
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
