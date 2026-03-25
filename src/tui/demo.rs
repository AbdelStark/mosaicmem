use crossterm::event::KeyCode;
use ratatui::prelude::*;
use ratatui::symbols::Marker;
use ratatui::widgets::canvas::{Canvas, Points};
use ratatui::widgets::*;

use super::theme;
use super::App;

const TOTAL_FRAMES: usize = 32;
const POINTS_PER_FRAME: usize = 24;

pub struct DemoState {
    pub visible_frames: usize,
    pub points: Vec<(f64, f64, usize)>, // (x, z, frame_idx)
    pub patches_data: Vec<u64>,
    pub tokens_data: Vec<u64>,
    pub coverage_data: Vec<u64>,
    pub points_data: Vec<u64>,
    pub running: bool,
}

impl Default for DemoState {
    fn default() -> Self {
        Self::new()
    }
}

impl DemoState {
    pub fn new() -> Self {
        let points = generate_point_cloud();
        let patches_data = generate_curve(1024, TOTAL_FRAMES, CurveShape::Sigmoid);
        let tokens_data = generate_curve(54000, TOTAL_FRAMES, CurveShape::Linear);
        let coverage_data = generate_curve(78, TOTAL_FRAMES, CurveShape::Sqrt);
        let points_data = generate_curve(12400, TOTAL_FRAMES, CurveShape::Linear);
        Self {
            visible_frames: 0,
            points,
            patches_data,
            tokens_data,
            coverage_data,
            points_data,
            running: true,
        }
    }
}

pub fn handle_key(state: &mut DemoState, key: KeyCode) {
    match key {
        KeyCode::Char(' ') => state.running = !state.running,
        KeyCode::Char('r') => {
            state.visible_frames = 0;
            state.running = true;
        }
        _ => {}
    }
}

pub fn render(f: &mut Frame, app: &mut App, area: Rect) {
    let state = &mut app.demo;

    // Advance animation
    if state.running && app.tick.is_multiple_of(4) {
        if state.visible_frames < TOTAL_FRAMES {
            state.visible_frames += 1;
        } else {
            // Pause briefly at end, then loop
            if app.tick.is_multiple_of(120) {
                state.visible_frames = 0;
            }
        }
    }

    let cols = Layout::horizontal([Constraint::Length(32), Constraint::Min(0)])
        .margin(1)
        .split(area);

    render_left_panel(f, app, cols[0]);
    render_right_panel(f, app, cols[1]);
}

fn render_left_panel(f: &mut Frame, app: &App, area: Rect) {
    let state = &app.demo;
    let vf = state.visible_frames;

    let rows = Layout::vertical([
        Constraint::Length(6),  // progress
        Constraint::Length(10), // memory stats
        Constraint::Length(7),  // config
        Constraint::Min(0),     // hint
    ])
    .split(area);

    // Progress
    let progress_block = Block::default()
        .title(Span::styled(" Progress ", theme::title_style()))
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(theme::border_style());

    let ratio = vf as f64 / TOTAL_FRAMES as f64;
    let progress_text = vec![
        Line::from(vec![
            Span::styled("  Window  ", Style::default().fg(theme::TEXT_MUTED)),
            Span::styled(
                format!("{}/4", (vf / 8).min(4)),
                theme::bold(theme::CYAN),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Frames  ", Style::default().fg(theme::TEXT_MUTED)),
            Span::styled(
                format!("{}/{}", vf, TOTAL_FRAMES),
                theme::bold(theme::CYAN),
            ),
        ]),
    ];
    f.render_widget(Paragraph::new(progress_text).block(progress_block), rows[0]);

    // Memory stats with gauges
    let mem_block = Block::default()
        .title(Span::styled(" Memory ", theme::title_style()))
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(theme::border_style());

    let patches = if vf > 0 { state.patches_data[vf - 1] } else { 0 };
    let tokens = if vf > 0 { state.tokens_data[vf - 1] } else { 0 };
    let coverage = if vf > 0 { state.coverage_data[vf - 1] } else { 0 };
    let points = if vf > 0 { state.points_data[vf - 1] } else { 0 };

    let mem_text = vec![
        Line::from(vec![
            Span::styled("  Patches  ", Style::default().fg(theme::TEXT_MUTED)),
            Span::styled(format!("{}", patches), theme::bold(theme::CYAN)),
            Span::styled("/1024", Style::default().fg(theme::DIM)),
        ]),
        Line::from(vec![
            Span::styled("  Points   ", Style::default().fg(theme::TEXT_MUTED)),
            Span::styled(format_num(points), theme::bold(theme::MAGENTA)),
        ]),
        Line::from(vec![
            Span::styled("  Tokens   ", Style::default().fg(theme::TEXT_MUTED)),
            Span::styled(format_num(tokens), theme::bold(theme::LAVENDER)),
        ]),
        Line::from(vec![
            Span::styled("  Coverage ", Style::default().fg(theme::TEXT_MUTED)),
            Span::styled(format!("{}%", coverage), theme::bold(theme::EMERALD)),
        ]),
        Line::default(),
        Line::from(Span::styled(
            format!("  {}", progress_bar(ratio, 24)),
            Style::default().fg(theme::gradient(theme::CYAN, theme::EMERALD, ratio)),
        )),
    ];
    f.render_widget(Paragraph::new(mem_text).block(mem_block), rows[1]);

    // Config
    let cfg_block = Block::default()
        .title(Span::styled(" Config ", theme::title_style()))
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(theme::border_style());

    let cfg_text = vec![
        Line::from(vec![
            Span::styled("  Resolution ", Style::default().fg(theme::TEXT_MUTED)),
            Span::styled("64x64", Style::default().fg(theme::TEXT)),
        ]),
        Line::from(vec![
            Span::styled("  Steps      ", Style::default().fg(theme::TEXT_MUTED)),
            Span::styled("5", Style::default().fg(theme::TEXT)),
        ]),
        Line::from(vec![
            Span::styled("  Windows    ", Style::default().fg(theme::TEXT_MUTED)),
            Span::styled("16f / 4 overlap", Style::default().fg(theme::TEXT)),
        ]),
    ];
    f.render_widget(Paragraph::new(cfg_text).block(cfg_block), rows[2]);

    // Hint
    let running = app.demo.running;
    f.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(" Space", Style::default().fg(theme::GOLD)),
            Span::styled(
                if running { " pause " } else { " play  " },
                Style::default().fg(theme::SLATE),
            ),
            Span::styled("r", Style::default().fg(theme::GOLD)),
            Span::styled(" reset", Style::default().fg(theme::SLATE)),
        ])),
        rows[3],
    );
}

fn render_right_panel(f: &mut Frame, app: &App, area: Rect) {
    let state = &app.demo;

    let rows = Layout::vertical([Constraint::Min(0), Constraint::Length(10)])
        .split(area);

    // Point cloud canvas
    render_canvas(f, app, rows[0]);

    // Sparklines
    render_sparklines(f, state, rows[1]);
}

fn render_canvas(f: &mut Frame, app: &App, area: Rect) {
    let state = &app.demo;
    let vf = state.visible_frames;

    let block = Block::default()
        .title(Line::from(vec![
            Span::styled(" Point Cloud ", theme::title_style()),
            Span::styled("top-down XZ ", Style::default().fg(theme::SLATE)),
        ]))
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(theme::border_style());

    // Collect visible points grouped by color bands
    let bands: usize = 8;
    let mut band_points: Vec<Vec<(f64, f64)>> = vec![Vec::new(); bands];

    for &(x, z, frame_idx) in &state.points {
        if frame_idx < vf {
            let band = (frame_idx * bands / TOTAL_FRAMES).min(bands - 1);
            band_points[band].push((x, z));
        }
    }

    // Camera position indicator
    let cam_angle = if vf > 0 {
        2.0 * std::f64::consts::PI * (vf - 1) as f64 / TOTAL_FRAMES as f64
    } else {
        0.0
    };
    let cam_x = 5.0 * cam_angle.cos();
    let cam_z = 5.0 * cam_angle.sin();

    let canvas = Canvas::default()
        .block(block)
        .x_bounds([-12.0, 12.0])
        .y_bounds([-12.0, 12.0])
        .marker(Marker::Braille)
        .paint(move |ctx| {
            // Draw point cloud bands with gradient colors
            for (i, pts) in band_points.iter().enumerate() {
                if pts.is_empty() {
                    continue;
                }
                let t = i as f64 / (bands - 1) as f64;
                let color = theme::gradient(theme::CYAN, theme::MAGENTA, t);
                ctx.draw(&Points {
                    coords: pts,
                    color,
                });
            }

            // Draw camera position
            if vf > 0 {
                ctx.draw(&Points {
                    coords: &[(cam_x, cam_z)],
                    color: theme::GOLD,
                });
                // Camera frustum indicator (small triangle)
                let fwd_x = cam_x + 1.5 * cam_angle.cos();
                let fwd_z = cam_z + 1.5 * cam_angle.sin();
                ctx.draw(&Points {
                    coords: &[(fwd_x, fwd_z)],
                    color: theme::GOLD,
                });
            }

            // Origin crosshair
            ctx.draw(&Points {
                coords: &[(0.0, 0.0)],
                color: theme::DIM,
            });
        });

    f.render_widget(canvas, area);
}

fn render_sparklines(f: &mut Frame, state: &DemoState, area: Rect) {
    let vf = state.visible_frames;
    let cols = Layout::horizontal([
        Constraint::Ratio(1, 4),
        Constraint::Ratio(1, 4),
        Constraint::Ratio(1, 4),
        Constraint::Ratio(1, 4),
    ])
    .split(area);

    let sparklines: Vec<(&str, &[u64], Color)> = vec![
        ("Patches", &state.patches_data, theme::CYAN),
        ("Points", &state.points_data, theme::MAGENTA),
        ("Tokens", &state.tokens_data, theme::LAVENDER),
        ("Coverage", &state.coverage_data, theme::EMERALD),
    ];

    for (i, (name, data, color)) in sparklines.into_iter().enumerate() {
        let slice = if vf > 0 { &data[..vf] } else { &data[..1] };
        let block = Block::default()
            .title(Span::styled(format!(" {name} "), theme::bold(color)))
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(theme::border_style());

        let sparkline = Sparkline::default()
            .block(block)
            .data(slice)
            .style(Style::default().fg(color));

        f.render_widget(sparkline, cols[i]);
    }
}

// ── Data Generation ──────────────────────────────────────────

enum CurveShape {
    Sigmoid,
    Linear,
    Sqrt,
}

fn generate_curve(max: u64, count: usize, shape: CurveShape) -> Vec<u64> {
    (0..count)
        .map(|i| {
            let t = if count > 1 {
                i as f64 / (count - 1) as f64
            } else {
                1.0
            };
            let v = match shape {
                CurveShape::Sigmoid => 1.0 / (1.0 + (-12.0 * (t - 0.4)).exp()),
                CurveShape::Linear => t,
                CurveShape::Sqrt => t.sqrt(),
            };
            (max as f64 * v) as u64
        })
        .collect()
}

fn generate_point_cloud() -> Vec<(f64, f64, usize)> {
    let mut points = Vec::new();
    let radius = 5.0;

    for frame in 0..TOTAL_FRAMES {
        let angle =
            2.0 * std::f64::consts::PI * frame as f64 / TOTAL_FRAMES as f64;
        let cam_x = radius * angle.cos();
        let cam_z = radius * angle.sin();

        for i in 0..POINTS_PER_FRAME {
            let depth = 1.5 + 3.5 * (i as f64 / POINTS_PER_FRAME as f64);
            // Deterministic pseudo-random spread
            let hash1 = ((frame * 7919 + i * 104729) % 10000) as f64 / 10000.0 - 0.5;
            let hash2 = ((frame * 7307 + i * 87211) % 10000) as f64 / 10000.0 - 0.5;
            let spread = 2.0;
            let px = cam_x + depth * angle.cos() + hash1 * spread;
            let pz = cam_z + depth * angle.sin() + hash2 * spread;
            points.push((px, pz, frame));
        }
    }
    points
}

fn progress_bar(ratio: f64, width: usize) -> String {
    let filled = (ratio * width as f64).round() as usize;
    let empty = width.saturating_sub(filled);
    format!("{}{}",
        "█".repeat(filled),
        "░".repeat(empty),
    )
}

fn format_num(n: u64) -> String {
    if n >= 1000 {
        format!("{:.1}k", n as f64 / 1000.0)
    } else {
        n.to_string()
    }
}
