use ratatui::prelude::*;
use ratatui::widgets::*;

use super::runner::BenchResult;
use super::theme;

#[derive(Default)]
pub struct BenchState {
    pub result: Option<BenchResult>,
    pub anim_iter: usize,
    pub log: Vec<String>,
}

impl BenchState {
    pub fn init(&mut self) {
        self.log
            .push("$ mosaicmem bench --num-frames 32 --steps 5 --iterations 5".into());
        self.log.push("Running benchmark...".into());
        let result = super::runner::run_bench();
        for (i, it) in result.iterations.iter().enumerate() {
            self.log.push(format!(
                "Iteration {}: {:.2}ms ({} patches, {} points)",
                i + 1,
                it.duration_ms,
                it.num_patches,
                it.num_points
            ));
        }
        self.log.push(format!("Average: {:.2}ms", result.avg_ms));
        self.log
            .push(format!("Throughput: {:.1} frames/sec", result.fps));
        self.result = Some(result);
    }
}

pub fn render(f: &mut Frame, state: &mut BenchState, area: Rect, tick: u64) {
    // Reveal iterations over time
    if tick.is_multiple_of(10)
        && let Some(ref result) = state.result
        && state.anim_iter < result.iterations.len()
    {
        state.anim_iter += 1;
    }

    let rows = Layout::vertical([
        Constraint::Length(3),  // command
        Constraint::Min(0),     // main content
        Constraint::Length(10), // log
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
            Span::styled("mosaicmem bench", theme::bold(theme::TEXT)),
            Span::styled(
                " --num-frames 32 --steps 5 --iterations 5",
                Style::default().fg(theme::TEXT_MUTED),
            ),
        ]))
        .block(cmd_block),
        rows[0],
    );

    render_main(f, state, rows[1], tick);
    render_log(f, state, rows[2]);
}

fn render_main(f: &mut Frame, state: &BenchState, area: Rect, tick: u64) {
    let cols = Layout::horizontal([Constraint::Min(0), Constraint::Length(36)]).split(area);

    render_chart(f, state, cols[0], tick);
    render_metrics(f, state, cols[1], tick);
}

fn render_chart(f: &mut Frame, state: &BenchState, area: Rect, _tick: u64) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(theme::border_style())
        .title(Span::styled(
            " Latency per Iteration ",
            theme::title_style(),
        ));

    let inner = block.inner(area);
    f.render_widget(block, area);

    let result = match &state.result {
        Some(r) => r,
        None => return,
    };

    if inner.height < 3 || inner.width < 6 {
        return;
    }

    let visible = state.anim_iter.min(result.iterations.len());
    let max_ms = result.max_ms * 1.2;
    let chart_h = inner.height.saturating_sub(2) as usize;
    let bar_width = ((inner.width as usize).saturating_sub(2)) / result.iterations.len().max(1);
    let bar_width = bar_width.max(3);

    let mut lines: Vec<Line> = Vec::new();

    for row in 0..chart_h {
        let threshold = max_ms * (1.0 - (row as f64 + 1.0) / chart_h as f64);
        let mut spans = vec![Span::raw(" ")];

        for (i, it) in result.iterations.iter().enumerate() {
            let is_visible = i < visible;
            let fill = if is_visible && it.duration_ms > threshold {
                "█"
            } else {
                " "
            };

            let color = if !is_visible {
                theme::DIM
            } else {
                // Color by relative speed: faster = greener
                let t = (it.duration_ms - result.min_ms) / (result.max_ms - result.min_ms + 0.001);
                theme::gradient(theme::EMERALD, theme::CORAL, t)
            };

            let bar = fill.repeat(bar_width.saturating_sub(1));
            spans.push(Span::styled(bar, Style::default().fg(color)));
            spans.push(Span::raw(" "));
        }
        lines.push(Line::from(spans));
    }

    // Labels
    let mut label_spans = vec![Span::raw(" ")];
    for (i, it) in result.iterations.iter().enumerate() {
        let label = if i < visible {
            format!("{:>w$.1}ms", it.duration_ms, w = bar_width - 1)
        } else {
            " ".repeat(bar_width - 1)
        };
        label_spans.push(Span::styled(label, Style::default().fg(theme::SLATE)));
        label_spans.push(Span::raw(" "));
    }
    lines.push(Line::from(label_spans));

    f.render_widget(Paragraph::new(lines), inner);
}

fn render_metrics(f: &mut Frame, state: &BenchState, area: Rect, tick: u64) {
    let rows = Layout::vertical([
        Constraint::Length(7),  // big number
        Constraint::Length(10), // stats
        Constraint::Min(0),     // sparkline
    ])
    .split(area);

    // Big throughput number
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(theme::breathe(theme::EMERALD, tick, 0.05)))
        .title(Span::styled(" Throughput ", theme::title_style()));

    let fps_text = if let Some(ref r) = state.result {
        vec![
            Line::default(),
            Line::from(Span::styled(
                format!("  {:.0}", r.fps),
                Style::default()
                    .fg(theme::EMERALD)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(Span::styled(
                "  frames/sec",
                Style::default().fg(theme::TEXT_MUTED),
            )),
        ]
    } else {
        vec![Line::from(Span::styled(
            "  ...",
            Style::default().fg(theme::SLATE),
        ))]
    };
    f.render_widget(Paragraph::new(fps_text).block(block), rows[0]);

    // Stats table
    let stats_block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(theme::border_style())
        .title(Span::styled(" Results ", theme::title_style()));

    let lines = if let Some(ref r) = state.result {
        vec![
            Line::default(),
            stat_line("Average", &format!("{:.2}ms", r.avg_ms), theme::CYAN),
            stat_line("Min", &format!("{:.2}ms", r.min_ms), theme::EMERALD),
            stat_line("Max", &format!("{:.2}ms", r.max_ms), theme::CORAL),
            stat_line(
                "Per-frame",
                &format!("{:.2}ms", r.per_frame_ms),
                theme::LAVENDER,
            ),
            Line::default(),
            stat_line("Frames", &format!("{}", r.num_frames), theme::TEXT),
            stat_line(
                "Iterations",
                &format!("{}", r.iterations.len()),
                theme::TEXT,
            ),
        ]
    } else {
        vec![Line::from(Span::styled(
            "  Running...",
            Style::default().fg(theme::SLATE),
        ))]
    };
    f.render_widget(Paragraph::new(lines).block(stats_block), rows[1]);

    // Latency sparkline
    if let Some(ref r) = state.result {
        let data: Vec<u64> = r.iterations[..state.anim_iter.min(r.iterations.len())]
            .iter()
            .map(|i| (i.duration_ms * 100.0) as u64)
            .collect();

        if !data.is_empty() && rows[2].height >= 4 {
            let sparkline = Sparkline::default()
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .border_type(BorderType::Rounded)
                        .border_style(theme::border_style())
                        .title(Span::styled(" Latency ", theme::title_style())),
                )
                .data(&data)
                .style(Style::default().fg(theme::PEACH));
            f.render_widget(sparkline, rows[2]);
        }
    }
}

fn render_log(f: &mut Frame, state: &BenchState, area: Rect) {
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
            } else if line.starts_with("Average") || line.starts_with("Throughput") {
                theme::bold(theme::GOLD)
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
