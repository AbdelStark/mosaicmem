use ratatui::prelude::*;
use ratatui::widgets::*;

use super::runner::OpsResult;
use super::theme;

#[derive(Default)]
pub struct OpsState {
    pub result: Option<OpsResult>,
    pub anim_step: usize,
    pub log: Vec<String>,
}


impl OpsState {
    pub fn init(&mut self) {
        self.log.push("$ mosaicmem demo  # then manipulate memory".into());
        self.log.push("Building memory store...".into());
        let result = super::runner::run_ops();
        self.log.push(format!("Original store: {} patches", result.original_patches));
        self.log.push(format!("manipulation::flip_vertical -> {} patches", result.flip_patches));
        self.log.push(format!("manipulation::erase_region(origin, r=2) -> {} patches", result.erase_patches));
        self.log.push(format!("manipulation::translate(+10x) -> {} patches", result.translate_patches));
        self.log.push(format!("manipulation::splice_horizontal -> {} patches", result.splice_patches));
        self.log.push(format!("Completed in {:.1}ms", result.elapsed_ms));
        self.result = Some(result);
    }
}

const OPS: [(&str, &str, &str); 5] = [
    ("Original", "Store built from 32-frame circular trajectory", "memory::store"),
    ("Flip Vertical", "Mirror all patch Y coordinates across origin", "manipulation::flip_vertical"),
    ("Erase Region", "Remove patches within sphere(origin, r=2)", "manipulation::erase_region"),
    ("Translate +10x", "Shift all patches by (10, 0, 0) in world space", "manipulation::translate"),
    ("Splice Horizontal", "Merge original + translated with 10u offset", "manipulation::splice_horizontal"),
];

pub fn render(f: &mut Frame, state: &mut OpsState, area: Rect, tick: u64) {
    // Reveal operations over time
    if tick.is_multiple_of(20) && state.anim_step < OPS.len() {
        state.anim_step += 1;
    }

    let rows = Layout::vertical([
        Constraint::Length(3),  // command
        Constraint::Min(0),     // operations
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
            Span::styled("mosaicmem demo", theme::bold(theme::TEXT)),
            Span::styled(" → manipulation::flip_vertical, erase_region, translate, splice_horizontal",
                Style::default().fg(theme::TEXT_MUTED)),
        ]))
        .block(cmd_block),
        rows[0],
    );

    render_operations(f, state, rows[1], tick);
    render_log(f, state, rows[2]);
}

fn render_operations(f: &mut Frame, state: &OpsState, area: Rect, _tick: u64) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(theme::border_style())
        .title(Line::from(vec![
            Span::styled(" Memory Operations ", theme::title_style()),
            Span::styled(
                format!(" {}/{} ", state.anim_step.min(OPS.len()), OPS.len()),
                Style::default().fg(theme::SLATE),
            ),
        ]));

    let inner = block.inner(area);
    f.render_widget(block, area);

    if inner.height < 4 { return; }

    let result = match &state.result {
        Some(r) => r,
        None => return,
    };

    let counts = [
        result.original_patches,
        result.flip_patches,
        result.erase_patches,
        result.translate_patches,
        result.splice_patches,
    ];

    let max_count = *counts.iter().max().unwrap_or(&1);

    // Layout: 5 operation cards
    let card_height = (inner.height as usize / OPS.len()).max(3);
    let mut y = inner.y;

    for (i, (name, desc, module)) in OPS.iter().enumerate() {
        let is_visible = i < state.anim_step;
        let remaining = inner.bottom().saturating_sub(y);
        if remaining < 3 { break; }

        let card_h = card_height.min(remaining as usize) as u16;
        let card_area = Rect::new(inner.x, y, inner.width, card_h);
        y += card_h;

        let count = counts[i];
        let bar_max_w = inner.width.saturating_sub(32) as usize;
        let bar_w = if max_count > 0 { count * bar_max_w / max_count } else { 0 };

        let colors = [theme::CYAN, theme::MAGENTA, theme::CORAL, theme::EMERALD, theme::GOLD];
        let color = colors[i % colors.len()];

        let icon = if is_visible { "◆" } else { "○" };
        let icon_color = if is_visible { color } else { theme::DIM };

        let mut lines: Vec<Line> = Vec::new();

        // Name + count + bar
        let bar_str = if is_visible {
            format!("{} {:>5} {}", "█".repeat(bar_w), count, "patches")
        } else {
            "...".to_string()
        };

        lines.push(Line::from(vec![
            Span::styled(format!(" {icon} "), Style::default().fg(icon_color)),
            Span::styled(format!("{name:20}"), if is_visible { theme::bold(theme::TEXT) } else { Style::default().fg(theme::DIM) }),
            Span::styled(bar_str, if is_visible { Style::default().fg(color) } else { Style::default().fg(theme::DIM) }),
        ]));

        if card_h > 1 {
            lines.push(Line::from(vec![
                Span::styled("    ", Style::default()),
                Span::styled(
                    if is_visible { format!("{desc}  ") } else { String::new() },
                    Style::default().fg(theme::TEXT_MUTED),
                ),
                Span::styled(
                    if is_visible { module.to_string() } else { String::new() },
                    Style::default().fg(theme::DIM),
                ),
            ]));
        }

        f.render_widget(Paragraph::new(lines), card_area);
    }
}

fn render_log(f: &mut Frame, state: &OpsState, area: Rect) {
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
            } else if line.contains("->") {
                Style::default().fg(theme::GOLD)
            } else {
                Style::default().fg(theme::TEXT_MUTED)
            };
            ListItem::new(Span::styled(format!("  {line}"), style))
        })
        .collect();

    let start = items.len().saturating_sub(area.height.saturating_sub(2) as usize);
    let visible: Vec<ListItem> = items.into_iter().skip(start).collect();
    f.render_widget(List::new(visible).block(block), area);
}
