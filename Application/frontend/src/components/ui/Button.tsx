import * as React from "react"
import { cn } from "@/lib/utils"

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
    variant?: "default" | "destructive" | "outline" | "secondary" | "ghost" | "link" | "dark" | "primary"
    size?: "default" | "sm" | "lg" | "icon"
    asChild?: boolean
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
    ({ className, variant = "default", size = "default", asChild = false, ...props }, ref) => {
        // Determine variant styles
        const variants = {
            default: "bg-primary text-primary-foreground hover:bg-primary/90",
            destructive: "bg-destructive text-destructive-foreground hover:bg-destructive/90",
            outline: "border border-input bg-background hover:bg-accent hover:text-accent-foreground text-foreground",
            secondary: "bg-secondary text-secondary-foreground hover:bg-secondary/80",
            ghost: "hover:bg-accent hover:text-accent-foreground text-muted-foreground hover:text-foreground",
            link: "text-primary underline-offset-4 hover:underline",
            dark: "bg-popover text-popover-foreground hover:bg-popover/80",
            primary: "bg-primary text-primary-foreground hover:bg-primary/90",
        }

        // Determine size styles
        const sizes = {
            default: "h-10 px-4 py-2",
            sm: "h-9 rounded-md px-3",
            lg: "h-11 rounded-md px-8",
            icon: "h-10 w-10",
        }

        // Base styles
        const baseStyles = "inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50"

        const variantClass = variants[variant] || variants.default
        const sizeClass = sizes[size] || sizes.default

        // Note: 'asChild' is ignored here as we removed the dependency on @radix-ui/react-slot.
        // If wrapping a component is strictly required, the user should handle it manually, but for now we render a button.

        return (
            <button
                className={cn(baseStyles, variantClass, sizeClass, className)}
                ref={ref}
                {...props}
            />
        )
    }
)
Button.displayName = "Button"

export { Button }
