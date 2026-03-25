use ratatui::prelude::*;
use ratatui::widgets::*;

use super::theme;
use super::App;

const LOGO: [&str; 6] = [
    r" ███╗   ███╗ ██████╗ ███████╗ █████╗ ██╗ ██████╗",
    r" ████╗ ████║██╔═══██╗██╔════╝██╔══██╗██║██╔════╝",
    r" ██╔████╔██║██║   ██║███████╗███████║██║██║     ",
    r" ██║╚██╔╝██║██║   ██║╚════██║██╔══██║██║██║     ",
    r" ██║ ╚═╝ ██║╚██████╔╝███████║██║  ██║██║╚██████╗",
    r" ╚═╝     ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝╚═╝ ╚═════╝",
];

pub fn render(f: &mut Frame, app: &App, area: Rect) {
    let layout = Layout::vertical([
        Constraint::Length(10), // logo + subtitle
        Constraint::Length(1),  // tagline
        Constraint::Length(1),  // gap
        Constraint::Min(0),     // content panels
    ])
    .split(area);

    render_logo(f, app, layout[0]);
    render_tagline(f, layout[1]);
    render_content(f, app, layout[3]);
}

fn render_logo(f: &mut Frame, app: &App, area: Rect) {
    let mut lines: Vec<Line> = vec![Line::default()];

    for (i, row) in LOGO.iter().enumerate() {
        let t_base = i as f64 / (LOGO.len() - 1) as f64;
        let shift = (app.tick as f64 * 0.015).sin() * 0.2;
        let t = (t_base + shift).clamp(0.0, 1.0);
        let from = theme::gradient(theme::CYAN, theme::MAGENTA, t);
        let to = theme::gradient(theme::MAGENTA, theme::GOLD, t);
        lines.push(theme::gradient_line(row, from, to));
    }

    lines.push(Line::default());
    let sub_t = ((app.tick as f64 * 0.02).sin() * 0.5 + 0.5).clamp(0.0, 1.0);
    lines.push(theme::gradient_line(
        "                        M  E  M",
        theme::gradient(theme::GOLD, theme::MAGENTA, sub_t),
        theme::gradient(theme::MAGENTA, theme::CYAN, sub_t),
    ));

    f.render_widget(
        Paragraph::new(Text::from(lines)).alignment(Alignment::Center),
        area,
    );
}

fn render_tagline(f: &mut Frame, area: Rect) {
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            "Spatial memory for camera-controlled video generation",
            Style::default().fg(theme::TEXT_MUTED),
        )))
        .alignment(Alignment::Center),
        area,
    );
}

fn render_content(f: &mut Frame, app: &App, area: Rect) {
    let cols = Layout::horizontal([Constraint::Percentage(38), Constraint::Percentage(62)])
        .margin(1)
        .split(area);

    render_left(f, app, cols[0]);
    render_right(f, app, cols[1]);
}

fn render_left(f: &mut Frame, _app: &App, area: Rect) {
    let rows = Layout::vertical([
        Constraint::Length(10),
        Constraint::Length(8),
        Constraint::Min(0),
    ])
    .split(area);

    // Overview stats
    let stats_block = Block::default()
        .title(Span::styled(" Overview ", theme::title_style()))
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(theme::border_style());

    let stats = Table::new(
        vec![
            stat_row("  Modules", "6", theme::CYAN),
            stat_row("  Source", "7,818 loc", theme::CYAN),
            stat_row("  Tests", "122", theme::EMERALD),
            stat_row("  Deps", "12 crates", theme::PEACH),
            stat_row("  Edition", "Rust 2024", theme::TEXT),
            stat_row("  Version", "0.1.0", theme::EMERALD),
        ],
        [Constraint::Length(12), Constraint::Min(10)],
    )
    .block(stats_block);
    f.render_widget(stats, rows[0]);

    // Links
    let links_block = Block::default()
        .title(Span::styled(" Links ", theme::title_style()))
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(theme::border_style());

    let links = Table::new(
        vec![
            stat_row("  Paper", "arXiv:2603.17117", theme::LAVENDER),
            stat_row("  Project", "mosaicmem.github.io", theme::LAVENDER),
            stat_row("  Demo", "YouTube", theme::CORAL),
            stat_row("  Crate", "crates.io/mosaicmem", theme::PEACH),
        ],
        [Constraint::Length(12), Constraint::Min(10)],
    )
    .block(links_block);
    f.render_widget(links, rows[1]);
}

fn render_right(f: &mut Frame, _app: &App, area: Rect) {
    let features: &[(&str, &str)] = &[
        ("Streaming 3D fusion", "incremental point cloud reconstruction"),
        ("kd-tree spatial index", "O(log n) retrieval via kiddo"),
        ("Warped RoPE", "geometry-aware rotary position encoding"),
        ("Warped Latent", "feature-level view alignment"),
        ("PRoPE", "progressive PE with temporal decay"),
        ("Autoregressive windowing", "arbitrary-length generation"),
        ("Adaptive keyframes", "motion-based frame selection"),
        ("Memory manipulation", "splice, transform, compose scenes"),
        ("Parallel computation", "multi-core via rayon"),
        ("Synthetic backends", "full pipeline without GPU/weights"),
        ("Zero unsafe code", "pure safe Rust throughout"),
    ];

    let items: Vec<ListItem> = features
        .iter()
        .enumerate()
        .map(|(i, (title, desc))| {
            let t = i as f64 / (features.len() - 1) as f64;
            let color = theme::gradient(theme::CYAN, theme::MAGENTA, t);
            ListItem::new(Line::from(vec![
                Span::styled("  ◆ ", Style::default().fg(color)),
                Span::styled(*title, theme::bold(theme::TEXT)),
                Span::styled(format!("  {desc}"), Style::default().fg(theme::TEXT_MUTED)),
            ]))
        })
        .collect();

    let block = Block::default()
        .title(Span::styled(" Key Features ", theme::title_style()))
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(theme::border_style())
        .padding(Padding::vertical(1));

    f.render_widget(List::new(items).block(block), area);
}

fn stat_row<'a>(label: &'a str, value: &'a str, color: Color) -> Row<'a> {
    Row::new(vec![
        Cell::from(Span::styled(label, Style::default().fg(theme::TEXT_MUTED))),
        Cell::from(Span::styled(value, theme::bold(color))),
    ])
}
