# Design System Strategy: AIA Agent 360

## Stitch Project
- **Project ID:** 10733830203648316936
- **Screen ID:** 27049691ec584f9893ac99253328bf26
- **Design System:** AIA Agent Obsidian
- **Generated:** 2026-03-27

## 1. Overview & Creative North Star: "The Obsidian Architect"
This design system adopts the **Obsidian Architect** philosophy. In the high-stakes world of insurance multi-agent systems, the UI feels like a precision instrument — authoritative, deep, and layered.

We use **Intentional Asymmetry** and **Tonal Depth** to guide the eye. By utilizing a hierarchy of dark surfaces rather than lines, we create a workspace that feels like a physical desk made of polished stone and glass. The brand's Crimson Red is used sparingly to denote human intent and brand authority.

## 2. Colors: The Depth Palette

### Surface Hierarchy (Nesting Principle)
| Token | Hex | Usage |
|-------|-----|-------|
| `surface` | `#0b1326` | Base background |
| `surface-container-lowest` | `#060e20` | Chat canvas |
| `surface-container-low` | `#131b2e` | Sidebars |
| `surface-container` | `#171f33` | Mid-level panels |
| `surface-container-high` | `#222a3d` | Thinking indicators |
| `surface-container-highest` | `#2d3449` | Active cards, AI bubbles |
| `surface-bright` | `#31394d` | Hover states |

### Accent Colors
| Token | Hex | Usage |
|-------|-----|-------|
| `primary-container` | `#c0392b` | Brand CTA, user bubbles |
| `primary` | `#ffb4a9` | Links, highlights |
| `secondary-container` | `#3131c0` | Multi-Tool agent |
| `secondary` | `#c0c1ff` | Code highlights |
| `tertiary-container` | `#007954` | Genie agent |
| `tertiary` | `#68dba9` | Success states |
| `on-surface` | `#dae2fd` | Primary text |
| `on-surface-variant` | `#94a3b8` | Muted text, labels |

### Agent-Specific Colors
| Agent | Background | Text |
|-------|-----------|------|
| Genie | `#007954` | `#68dba9` |
| Multi-Tool | `#3131c0` | `#c0c1ff` |
| Analysis | `#d97706` | `#d97706` |
| Visualization | `#c0392b` | `#ffb4a9` |

## 3. Typography: Inter
- **Headlines:** Inter 800, -0.02em tracking
- **Body:** Inter 400, line-height 1.6
- **Labels:** Inter 700, uppercase, +0.12em tracking, 0.65em size
- **Code:** Monospace, `#c0c1ff` on `#171f33`

## 4. Component Patterns
- **Roundness:** 12px cards, 8px inputs, full-round avatars/pills
- **No explicit borders** — use tonal shifts for hierarchy
- **Ghost borders** at 15% opacity when needed
- **Glassmorphism** for floating elements (backdrop-blur: 12px)
- **Thinking indicator:** Pulsing glow animation, not spinner
