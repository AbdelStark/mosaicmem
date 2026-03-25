use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

// ── Glamour Palette ──────────────────────────────────────────

pub const CYAN: Color = Color::Rgb(0, 212, 255);
pub const MAGENTA: Color = Color::Rgb(200, 60, 255);
pub const GOLD: Color = Color::Rgb(255, 215, 0);
pub const EMERALD: Color = Color::Rgb(0, 255, 136);
pub const CORAL: Color = Color::Rgb(255, 107, 107);
pub const LAVENDER: Color = Color::Rgb(167, 139, 250);
pub const PEACH: Color = Color::Rgb(255, 182, 139);
pub const ROSE: Color = Color::Rgb(244, 114, 182);

pub const SURFACE: Color = Color::Rgb(12, 14, 24);
pub const SURFACE_1: Color = Color::Rgb(20, 26, 42);
pub const SURFACE_2: Color = Color::Rgb(28, 36, 56);
pub const SLATE: Color = Color::Rgb(100, 116, 139);
pub const DIM: Color = Color::Rgb(40, 50, 68);
pub const TEXT: Color = Color::Rgb(226, 232, 240);
pub const TEXT_MUTED: Color = Color::Rgb(148, 163, 184);

// ── Gradient ─────────────────────────────────────────────────

pub fn gradient(from: Color, to: Color, t: f64) -> Color {
    let t = t.clamp(0.0, 1.0);
    let (r1, g1, b1) = rgb(from);
    let (r2, g2, b2) = rgb(to);
    Color::Rgb(lerp_u8(r1, r2, t), lerp_u8(g1, g2, t), lerp_u8(b1, b2, t))
}

pub fn gradient_line(text: &str, from: Color, to: Color) -> Line<'static> {
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    if len == 0 {
        return Line::default();
    }
    Line::from(
        chars
            .into_iter()
            .enumerate()
            .map(|(i, c)| {
                let t = if len > 1 { i as f64 / (len - 1) as f64 } else { 0.0 };
                Span::styled(
                    c.to_string(),
                    Style::default().fg(gradient(from, to, t)).add_modifier(Modifier::BOLD),
                )
            })
            .collect::<Vec<_>>(),
    )
}

pub fn breathe(base: Color, tick: u64, speed: f64) -> Color {
    let phase = ((tick as f64 * speed).sin() * 0.25 + 0.75).clamp(0.5, 1.0);
    let (r, g, b) = rgb(base);
    Color::Rgb(
        (r as f64 * phase) as u8,
        (g as f64 * phase) as u8,
        (b as f64 * phase) as u8,
    )
}

pub fn bold(color: Color) -> Style {
    Style::default().fg(color).add_modifier(Modifier::BOLD)
}

pub fn title_style() -> Style {
    Style::default().fg(CYAN).add_modifier(Modifier::BOLD)
}

pub fn border_style() -> Style {
    Style::default().fg(DIM)
}

pub fn cmd_style() -> Style {
    Style::default().fg(EMERALD)
}

fn rgb(c: Color) -> (u8, u8, u8) {
    match c { Color::Rgb(r, g, b) => (r, g, b), _ => (255, 255, 255) }
}

fn lerp_u8(a: u8, b: u8, t: f64) -> u8 {
    (a as f64 + (b as f64 - a as f64) * t).round() as u8
}
