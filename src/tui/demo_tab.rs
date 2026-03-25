use ratatui::prelude::*;
use ratatui::symbols::Marker;
use ratatui::widgets::canvas::{Canvas, Points};
use ratatui::widgets::*;

use super::runner::DemoResult;
use super::theme;

pub struct DemoState {
    pub result: Option<DemoResult>,
    pub anim_frame: usize,
    pub running: bool,
    pub log: Vec<String>,
}

impl Default for DemoState {
    fn default() -> Self {
        Self { result: None, anim_frame: 0, running: true, log: Vec::new() }
    }
}

impl DemoState {
    pub fn init(&mut self) {
        self.log.push("$ mosaicmem demo --num-frames 32 --width 64 --height 64 --steps 5".into());
        self.log.push("Running pipeline...".into());
        let result = super::runner::run_demo();
        self.log.push(format!("Created circular trajectory: {} poses", result.num_frames));
        self.log.push(format!("Path length: {:.2} units", result.path_length));
        self.log.push(format!("Generated {} windows, {} total values", result.num_windows, result.total_values));
        self.log.push(format!(
            "Final: {} patches, {} points, {} keyframes, {} tokens",
            result.num_patches, result.num_points, result.num_keyframes, result.total_tokens
        ));
        self.log.push(format!("Pipeline completed in {:.1}ms", result.elapsed_ms));
        self.log.push("--- Memory Manipulation ---".into());
        self.log.push(format!("flip_vertical: {} patches", result.flip_patches));
        self.log.push(format!("erase_region(origin, r=2): {} patches remaining", result.erase_patches));
        self.log.push(format!("translate(+10, 0, 0): {} patches", result.translate_patches));
        self.result = Some(result);
    }
}

pub fn render(f: &mut Frame, state: &mut DemoState, area: Rect, tick: u64) {
    // Advance animation
    if state.running && tick.is_multiple_of(2)
        && let Some(ref result) = state.result {
            let max = result.cloud_xz.len();
            if state.anim_frame < max {
                state.anim_frame += (max / 40).max(1);
                state.anim_frame = state.anim_frame.min(max);
            }
        }

    let cols = Layout::horizontal([Constraint::Min(0), Constraint::Length(38)])
        .split(area);

    render_main(f, state, cols[0], tick);
    render_sidebar(f, state, cols[1], tick);
}

fn render_main(f: &mut Frame, state: &DemoState, area: Rect, tick: u64) {
    let rows = Layout::vertical([
        Constraint::Length(3),  // command
        Constraint::Min(0),     // canvas
        Constraint::Length(10), // log
    ])
    .split(area);

    // Command display
    let cmd_block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(theme::border_style())
        .title(Span::styled(" Command ", theme::title_style()));

    let cmd_text = Line::from(vec![
        Span::styled("$ ", Style::default().fg(theme::EMERALD)),
        Span::styled("mosaicmem demo", theme::bold(theme::TEXT)),
        Span::styled(" --num-frames 32 --width 64 --height 64 --steps 5", Style::default().fg(theme::TEXT_MUTED)),
    ]);
    f.render_widget(Paragraph::new(cmd_text).block(cmd_block), rows[0]);

    // Point cloud canvas
    render_canvas(f, state, rows[1], tick);

    // Log output
    render_log(f, state, rows[2]);
}

fn render_canvas(f: &mut Frame, state: &DemoState, area: Rect, _tick: u64) {
    let result = match &state.result {
        Some(r) => r,
        None => {
            let block = Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(theme::border_style())
                .title(Span::styled(" Point Cloud ", theme::title_style()));
            f.render_widget(
                Paragraph::new(Span::styled("  Initializing...", Style::default().fg(theme::SLATE)))
                    .block(block),
                area,
            );
            return;
        }
    };

    let visible = state.anim_frame;
    let total = result.cloud_xz.len();
    let progress = if total > 0 { visible as f64 / total as f64 } else { 0.0 };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(theme::gradient(theme::CYAN, theme::MAGENTA, progress)))
        .title(Line::from(vec![
            Span::styled(" Point Cloud ", theme::title_style()),
            Span::styled(
                format!("{}/{} pts ", visible, total),
                Style::default().fg(theme::SLATE),
            ),
            Span::styled(
                format!("{}%", (progress * 100.0) as u32),
                theme::bold(theme::gradient(theme::CYAN, theme::EMERALD, progress)),
            ),
            Span::raw(" "),
        ]));

    // Split visible points into color bands
    let bands: usize = 6;
    let mut band_points: Vec<Vec<(f64, f64)>> = vec![Vec::new(); bands];
    for (i, &(x, z)) in result.cloud_xz.iter().enumerate() {
        if i >= visible { break; }
        let band = (i * bands / total.max(1)).min(bands - 1);
        band_points[band].push((x, z));
    }

    // Camera positions
    let cam_pts: Vec<(f64, f64)> = result
        .cam_positions
        .iter()
        .map(|p| (p[0] as f64, p[2] as f64))
        .collect();

    // Current camera
    let cam_idx = (visible * result.cam_positions.len() / total.max(1))
        .min(result.cam_positions.len().saturating_sub(1));
    let cam_now = if !result.cam_positions.is_empty() {
        let p = &result.cam_positions[cam_idx];
        Some((p[0] as f64, p[2] as f64))
    } else {
        None
    };

    let canvas = Canvas::default()
        .block(block)
        .x_bounds([-12.0, 12.0])
        .y_bounds([-12.0, 12.0])
        .marker(Marker::Braille)
        .paint(move |ctx| {
            // Camera trajectory (dim)
            ctx.draw(&Points { coords: &cam_pts, color: theme::DIM });

            // Point cloud bands
            for (i, pts) in band_points.iter().enumerate() {
                if pts.is_empty() { continue; }
                let t = i as f64 / (bands - 1) as f64;
                ctx.draw(&Points {
                    coords: pts,
                    color: theme::gradient(theme::CYAN, theme::MAGENTA, t),
                });
            }

            // Current camera position (bright)
            if let Some(pos) = cam_now {
                ctx.draw(&Points { coords: &[pos], color: theme::GOLD });
            }

            // Origin marker
            ctx.draw(&Points { coords: &[(0.0, 0.0)], color: theme::SLATE });
        });

    f.render_widget(canvas, area);
}

fn render_log(f: &mut Frame, state: &DemoState, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(theme::border_style())
        .title(Span::styled(" Output ", theme::title_style()));

    let items: Vec<ListItem> = state
        .log
        .iter()
        .map(|line| {
            let style = if line.starts_with('$') {
                theme::cmd_style()
            } else if line.starts_with("---") {
                theme::bold(theme::GOLD)
            } else {
                Style::default().fg(theme::TEXT_MUTED)
            };
            ListItem::new(Span::styled(format!("  {line}"), style))
        })
        .collect();

    let visible_start = items.len().saturating_sub((area.height as usize).saturating_sub(2));
    let visible_items: Vec<ListItem> = items.into_iter().skip(visible_start).collect();
    f.render_widget(List::new(visible_items).block(block), area);
}

fn render_sidebar(f: &mut Frame, state: &DemoState, area: Rect, tick: u64) {
    let rows = Layout::vertical([
        Constraint::Length(12), // stats
        Constraint::Length(8),  // memory ops
        Constraint::Length(8),  // config
        Constraint::Min(0),     // sparkline area
    ])
    .split(area);

    render_stats(f, state, rows[0], tick);
    render_ops(f, state, rows[1]);
    render_config(f, rows[2]);
    render_sparkline_panel(f, state, rows[3], tick);
}

fn render_stats(f: &mut Frame, state: &DemoState, area: Rect, _tick: u64) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(theme::border_style())
        .title(Span::styled(" Live Stats ", theme::title_style()));

    let (patches, points, keyframes, tokens, elapsed) = state
        .result
        .as_ref()
        .map(|r| (r.num_patches, r.num_points, r.num_keyframes, r.total_tokens, r.elapsed_ms))
        .unwrap_or((0, 0, 0, 0, 0.0));

    let anim_t = state
        .result
        .as_ref()
        .map(|r| {
            let total = r.cloud_xz.len();
            if total > 0 { state.anim_frame as f64 / total as f64 } else { 1.0 }
        })
        .unwrap_or(0.0);

    let animate = |v: usize| -> String {
        let animated = (v as f64 * anim_t) as usize;
        format_num(animated)
    };

    let lines = vec![
        Line::default(),
        metric_line("  Patches  ", &animate(patches), theme::CYAN),
        metric_line("  Points   ", &animate(points), theme::MAGENTA),
        metric_line("  Keyframes", &format!("{}", (keyframes as f64 * anim_t) as usize), theme::GOLD),
        metric_line("  Tokens   ", &animate(tokens), theme::LAVENDER),
        Line::default(),
        metric_line("  Windows  ", &format!("{}", state.result.as_ref().map(|r| r.num_windows).unwrap_or(0)), theme::PEACH),
        metric_line("  Values   ", &animate(state.result.as_ref().map(|r| r.total_values).unwrap_or(0)), theme::ROSE),
        Line::from(vec![
            Span::styled("  Time     ", Style::default().fg(theme::TEXT_MUTED)),
            Span::styled(format!("{:.1}ms", elapsed), theme::bold(theme::EMERALD)),
        ]),
    ];

    f.render_widget(Paragraph::new(lines).block(block), area);
}

fn render_ops(f: &mut Frame, state: &DemoState, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(theme::border_style())
        .title(Span::styled(" Memory Ops ", theme::title_style()));

    let lines = if let Some(ref r) = state.result {
        vec![
            Line::default(),
            op_line("flip_vertical", r.flip_patches, theme::CYAN),
            op_line("erase(r=2)", r.erase_patches, theme::CORAL),
            op_line("translate +10x", r.translate_patches, theme::EMERALD),
        ]
    } else {
        vec![Line::from(Span::styled("  waiting...", Style::default().fg(theme::SLATE)))]
    };

    f.render_widget(Paragraph::new(lines).block(block), area);
}

fn render_config(f: &mut Frame, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(theme::border_style())
        .title(Span::styled(" Config ", theme::title_style()));

    let lines = vec![
        Line::default(),
        cfg_line("Resolution", "64x64"),
        cfg_line("Steps", "5"),
        cfg_line("Windows", "16f / 2 overlap"),
        cfg_line("Trajectory", "circle r=5"),
    ];

    f.render_widget(Paragraph::new(lines).block(block), area);
}

fn render_sparkline_panel(f: &mut Frame, state: &DemoState, area: Rect, _tick: u64) {
    if area.height < 4 { return; }

    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(theme::border_style())
        .title(Span::styled(" Cloud Growth ", theme::title_style()));

    // Build sparkline from progressive point counts
    if let Some(ref result) = state.result {
        let total = result.cloud_xz.len();
        let steps = (area.width as usize).saturating_sub(4).min(40);
        let data: Vec<u64> = (0..steps)
            .map(|i| {
                let frame = (i + 1) * state.anim_frame / steps.max(1);
                frame.min(total) as u64
            })
            .collect();

        let sparkline = Sparkline::default()
            .block(block)
            .data(&data)
            .style(Style::default().fg(theme::CYAN));
        f.render_widget(sparkline, area);
    } else {
        f.render_widget(block, area);
    }
}

// ── Helpers ──────────────────────────────────────────────────

fn metric_line<'a>(label: &'a str, value: &str, color: Color) -> Line<'a> {
    Line::from(vec![
        Span::styled(label, Style::default().fg(theme::TEXT_MUTED)),
        Span::styled(format!(" {value}"), theme::bold(color)),
    ])
}

fn op_line(name: &str, count: usize, color: Color) -> Line<'_> {
    Line::from(vec![
        Span::styled(format!("  {name}: "), Style::default().fg(theme::TEXT_MUTED)),
        Span::styled(format!("{count}"), theme::bold(color)),
        Span::styled(" patches", Style::default().fg(theme::DIM)),
    ])
}

fn cfg_line<'a>(key: &'a str, val: &'a str) -> Line<'a> {
    Line::from(vec![
        Span::styled(format!("  {key:12}"), Style::default().fg(theme::TEXT_MUTED)),
        Span::styled(val, Style::default().fg(theme::TEXT)),
    ])
}

fn format_num(n: usize) -> String {
    if n >= 1_000_000 { format!("{:.1}M", n as f64 / 1_000_000.0) }
    else if n >= 1_000 { format!("{:.1}k", n as f64 / 1_000.0) }
    else { n.to_string() }
}
