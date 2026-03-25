use ratatui::prelude::*;
use ratatui::widgets::*;

use super::runner::CoverageResult;
use super::theme;

#[derive(Default)]
pub struct CoverageState {
    pub result: Option<CoverageResult>,
    pub anim_frame: usize,
    pub log: Vec<String>,
}

impl CoverageState {
    pub fn init(&mut self) {
        self.log
            .push("$ mosaicmem inspect --trajectory traj.json --coverage".into());
        self.log
            .push("Building synthetic memory from trajectory...".into());
        let result = super::runner::run_coverage();
        self.log.push(format!(
            "Stored {} patches, {} points",
            result.num_patches, result.num_points
        ));
        self.log
            .push(format!("Total tokens: {}", result.total_tokens));
        self.log.push(format!(
            "Bounding box: {:.1} x {:.1} x {:.1}",
            result.bbox_size[0], result.bbox_size[1], result.bbox_size[2]
        ));
        let avg_cov: f32 = if result.frames.is_empty() {
            0.0
        } else {
            result.frames.iter().map(|f| f.coverage).sum::<f32>() / result.frames.len() as f32
        };
        self.log
            .push(format!("Average coverage: {:.1}%", avg_cov * 100.0));
        self.log
            .push(format!("Completed in {:.1}ms", result.elapsed_ms));
        self.result = Some(result);
    }
}

pub fn render(f: &mut Frame, state: &mut CoverageState, area: Rect, tick: u64) {
    // Animate: reveal one bar per tick
    if tick.is_multiple_of(2)
        && let Some(ref result) = state.result
        && state.anim_frame < result.frames.len()
    {
        state.anim_frame += 1;
    }

    let rows = Layout::vertical([
        Constraint::Length(3),  // command
        Constraint::Min(0),     // coverage chart
        Constraint::Length(12), // bottom: stats + log
    ])
    .split(area);

    // Command
    let cmd_block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(theme::border_style())
        .title(Span::styled(" Command ", theme::title_style()));

    f.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled("$ ", theme::cmd_style()),
            Span::styled("mosaicmem inspect", theme::bold(theme::TEXT)),
            Span::styled(
                " --trajectory traj.json --coverage",
                Style::default().fg(theme::TEXT_MUTED),
            ),
        ]))
        .block(cmd_block),
        rows[0],
    );

    // Coverage chart
    render_coverage_chart(f, state, rows[1], tick);

    // Bottom: stats + log
    let bottom_cols =
        Layout::horizontal([Constraint::Length(36), Constraint::Min(0)]).split(rows[2]);
    render_stats(f, state, bottom_cols[0]);
    render_log(f, state, bottom_cols[1]);
}

fn render_coverage_chart(f: &mut Frame, state: &CoverageState, area: Rect, _tick: u64) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(theme::border_style())
        .title(Line::from(vec![
            Span::styled(" Per-Frame Coverage ", theme::title_style()),
            Span::styled(
                format!(
                    " {}/{} frames ",
                    state.anim_frame,
                    state.result.as_ref().map(|r| r.frames.len()).unwrap_or(0)
                ),
                Style::default().fg(theme::SLATE),
            ),
        ]));

    let inner = block.inner(area);
    f.render_widget(block, area);

    if inner.height < 2 || inner.width < 4 {
        return;
    }

    let result = match &state.result {
        Some(r) => r,
        None => return,
    };

    let visible = state.anim_frame.min(result.frames.len());
    let chart_height = inner.height.saturating_sub(2) as usize;
    let chart_width = inner.width.saturating_sub(2) as usize;

    // Render bar chart: one column per frame (or group if too many)
    let num_frames = result.frames.len();
    let frames_per_col = (num_frames as f64 / chart_width as f64).ceil() as usize;
    let frames_per_col = frames_per_col.max(1);

    let mut lines: Vec<Line> = Vec::new();

    // Build column heights
    let mut cols: Vec<(f32, bool)> = Vec::new(); // (coverage, is_visible)
    for col_idx in 0..chart_width {
        let start = col_idx * frames_per_col;
        let end = (start + frames_per_col).min(num_frames);
        if start >= num_frames {
            break;
        }

        let avg_cov: f32 = result.frames[start..end]
            .iter()
            .map(|f| f.coverage)
            .sum::<f32>()
            / (end - start) as f32;

        let visible_in_range = start < visible;
        cols.push((avg_cov, visible_in_range));
    }

    // Render top-down
    for row in 0..chart_height {
        let threshold = 1.0 - (row as f32 + 1.0) / chart_height as f32;
        let mut spans = vec![Span::raw(" ")];
        for &(cov, vis) in &cols {
            if !vis {
                spans.push(Span::styled(" ", Style::default()));
            } else if cov > threshold {
                let t = cov as f64;
                let color = theme::gradient(theme::CORAL, theme::EMERALD, t);
                spans.push(Span::styled("█", Style::default().fg(color)));
            } else {
                spans.push(Span::styled("░", Style::default().fg(theme::DIM)));
            }
        }
        lines.push(Line::from(spans));
    }

    // Scale labels
    lines.push(Line::from(vec![
        Span::styled(" 0%", Style::default().fg(theme::DIM)),
        Span::styled(" ".repeat(chart_width.saturating_sub(8)), Style::default()),
        Span::styled("100%", Style::default().fg(theme::DIM)),
    ]));

    f.render_widget(Paragraph::new(lines), inner);
}

fn render_stats(f: &mut Frame, state: &CoverageState, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(theme::border_style())
        .title(Span::styled(" Stats ", theme::title_style()));

    let lines = if let Some(ref r) = state.result {
        let avg_cov = if r.frames.is_empty() {
            0.0
        } else {
            r.frames.iter().map(|f| f.coverage).sum::<f32>() / r.frames.len() as f32 * 100.0
        };
        let max_cov = r.frames.iter().map(|f| f.coverage).fold(0.0f32, f32::max) * 100.0;
        let min_cov = r.frames.iter().map(|f| f.coverage).fold(1.0f32, f32::min) * 100.0;
        vec![
            Line::default(),
            stat_line("Patches", &format!("{}", r.num_patches), theme::CYAN),
            stat_line("Points", &format!("{}", r.num_points), theme::MAGENTA),
            stat_line("Tokens", &format!("{}", r.total_tokens), theme::LAVENDER),
            Line::default(),
            stat_line("Avg Cov", &format!("{:.1}%", avg_cov), theme::EMERALD),
            stat_line("Max Cov", &format!("{:.1}%", max_cov), theme::GOLD),
            stat_line("Min Cov", &format!("{:.1}%", min_cov), theme::CORAL),
        ]
    } else {
        vec![Line::from(Span::styled(
            "  Loading...",
            Style::default().fg(theme::SLATE),
        ))]
    };

    f.render_widget(Paragraph::new(lines).block(block), area);
}

fn render_log(f: &mut Frame, state: &CoverageState, area: Rect) {
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
            } else {
                Style::default().fg(theme::TEXT_MUTED)
            };
            ListItem::new(Span::styled(format!("  {line}"), style))
        })
        .collect();

    let start = items
        .len()
        .saturating_sub(area.height.saturating_sub(2) as usize);
    let visible: Vec<ListItem> = items.into_iter().skip(start).collect();
    f.render_widget(List::new(visible).block(block), area);
}

fn stat_line<'a>(label: &'a str, value: &str, color: Color) -> Line<'a> {
    Line::from(vec![
        Span::styled(
            format!("  {label:10}"),
            Style::default().fg(theme::TEXT_MUTED),
        ),
        Span::styled(value.to_string(), theme::bold(color)),
    ])
}
