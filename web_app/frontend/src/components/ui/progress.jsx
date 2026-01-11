import * as React from "react"
import { cn } from "../../lib/utils"
import { motion } from "framer-motion"

// Animated Progress Bar
const Progress = React.forwardRef(({ 
  value = 0, 
  max = 100,
  color = "blue",
  size = "md",
  showLabel = false,
  animated = true,
  className, 
  ...props 
}, ref) => {
  const percentage = Math.min(100, Math.max(0, (value / max) * 100))
  
  const colorMap = {
    blue: "bg-blue-500",
    green: "bg-emerald-500",
    red: "bg-rose-500",
    yellow: "bg-amber-500",
    purple: "bg-violet-500",
    gradient: "bg-gradient-to-r from-blue-500 via-indigo-500 to-violet-500"
  }
  
  const sizeMap = {
    sm: "h-1.5",
    md: "h-2.5",
    lg: "h-4"
  }
  
  return (
    <div ref={ref} className={cn("w-full", className)} {...props}>
      <div className={cn(
        "w-full bg-gray-100 rounded-full overflow-hidden",
        sizeMap[size]
      )}>
        <motion.div
          initial={animated ? { width: 0 } : false}
          animate={{ width: `${percentage}%` }}
          transition={{ duration: 1, ease: "easeOut" }}
          className={cn(
            "h-full rounded-full",
            colorMap[color]
          )}
        />
      </div>
      {showLabel && (
        <div className="flex justify-between mt-1 text-xs text-gray-500">
          <span>{value}</span>
          <span>{max}</span>
        </div>
      )}
    </div>
  )
})
Progress.displayName = "Progress"

// Medical-style Threshold Progress (with zones)
const ThresholdProgress = React.forwardRef(({ 
  value = 0,
  thresholds = { low: 30, high: 70 },
  className,
  ...props 
}, ref) => {
  const getColor = () => {
    if (value >= thresholds.high) return "bg-emerald-500"
    if (value >= thresholds.low) return "bg-amber-400"
    return "bg-gray-300"
  }
  
  return (
    <div ref={ref} className={cn("relative w-full", className)} {...props}>
      <div className="h-3 w-full bg-gray-100 rounded-full overflow-hidden relative">
        {/* Threshold markers */}
        <div 
          className="absolute top-0 bottom-0 w-0.5 bg-gray-300 z-10" 
          style={{ left: `${thresholds.low}%` }} 
        />
        <div 
          className="absolute top-0 bottom-0 w-0.5 bg-gray-300 z-10" 
          style={{ left: `${thresholds.high}%` }} 
        />
        
        {/* Progress fill */}
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${value}%` }}
          transition={{ duration: 1, ease: "easeOut" }}
          className={cn("h-full rounded-full", getColor())}
        />
      </div>
      
      {/* Zone labels */}
      <div className="flex justify-between mt-1 text-[10px] text-gray-400 font-medium">
        <span>Low</span>
        <span>Ambiguous</span>
        <span>High</span>
      </div>
    </div>
  )
})
ThresholdProgress.displayName = "ThresholdProgress"

// Circular Progress (for confidence display)
const CircularProgress = React.forwardRef(({ 
  value = 0, 
  size = 80,
  strokeWidth = 8,
  color = "blue",
  children,
  className, 
  ...props 
}, ref) => {
  const radius = (size - strokeWidth) / 2
  const circumference = radius * 2 * Math.PI
  const offset = circumference - (value / 100) * circumference
  
  const colorMap = {
    blue: "stroke-blue-500",
    green: "stroke-emerald-500",
    red: "stroke-rose-500",
    purple: "stroke-violet-500"
  }
  
  return (
    <div ref={ref} className={cn("relative inline-flex", className)} {...props}>
      <svg width={size} height={size} className="transform -rotate-90">
        {/* Background circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="currentColor"
          strokeWidth={strokeWidth}
          className="text-gray-100"
        />
        {/* Progress circle */}
        <motion.circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset: offset }}
          transition={{ duration: 1.5, ease: "easeOut" }}
          style={{ strokeDasharray: circumference }}
          className={colorMap[color]}
        />
      </svg>
      {/* Center content */}
      <div className="absolute inset-0 flex items-center justify-center">
        {children || <span className="text-lg font-bold text-gray-900">{value}%</span>}
      </div>
    </div>
  )
})
CircularProgress.displayName = "CircularProgress"

export { Progress, ThresholdProgress, CircularProgress }
