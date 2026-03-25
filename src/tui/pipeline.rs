use crossterm::event::KeyCode;
use ratatui::prelude::*;
use ratatui::widgets::*;

use super::theme;
use super::App;

pub struct PipelineState {
    pub active: usize,
    pub paused: bool,
}

impl Default for PipelineState {
    fn default() -> Self {
        Self::new()
    }
}

impl PipelineState {
    pub fn new() -> Self {
        Self {
            active: 0,
            paused: false,
        }
    }
}

pub fn handle_key(state: &mut PipelineState, key: KeyCode) {
    match key {
        KeyCode::Char(' ') => state.paused = !state.paused,
        KeyCode::Right | KeyCode::Char('l') => state.active = (state.active + 1) % 6,
        KeyCode::Left | KeyCode::Char('h') => state.active = (state.active + 5) % 6,
        _ => {}
    }
}

struct Stage {
    name: &'static str,
    icon: &'static str,
    detail: &'static str,
    module: &'static str,
}

const STAGES: [Stage; 6] = [
    Stage {
        name: "Depth Estimate",
        icon: "◧",
        detail: "Per-pixel monocular depth estimation from keyframes. \
                 Provides geometric foundation for 3D reconstruction. \
                 The synthetic backend generates structured depth maps \
                 with realistic near-far distributions for testing.",
        module: "geometry::depth",
    },
    Stage {
        name: "Lift to 3D",
        icon: "△",
        detail: "Unproject image patches into world-space 3D points \
                 using camera intrinsics and estimated depth. Creates \
                 positioned patch representations in a shared coordinate \
                 frame via streaming fusion.",
        module: "geometry::fusion",
    },
    Stage {
        name: "Store Memory",
        icon: "▣",
        detail: "Insert 3D patches into the MosaicMemoryStore backed by \
                 a kd-tree spatial index (kiddo). Enables O(log n) \
                 nearest-neighbor queries over millions of patches with \
                 sub-millisecond retrieval latency.",
        module: "memory::store",
    },
    Stage {
        name: "Retrieve Patches",
        icon: "◎",
        detail: "Given a target camera pose, query the spatial memory \
                 for relevant patches. Applies frustum culling, temporal \
                 decay scoring, diversity-aware filtering, and depth \
                 sorting to select optimal conditioning context.",
        module: "memory::retrieval",
    },
    Stage {
        name: "Geometric Align",
        icon: "⬡",
        detail: "Apply Warped RoPE at the attention level to encode 3D \
                 spatial relationships into position encodings. Apply \
                 Warped Latent at the feature level to reproject patch \
                 content into the target view's latent space.",
        module: "attention::warped_rope",
    },
    Stage {
        name: "Diffuse & Generate",
        icon: "◈",
        detail: "Inject aligned memory context into the diffusion \
                 denoising loop via MemoryCrossAttention. PRoPE provides \
                 progressive temporal encoding. DDPM scheduler drives \
                 the denoising steps to produce geometry-consistent frames.",
        module: "pipeline::inference",
    },
];

pub fn render(f: &mut Frame, app: &mut App, area: Rect) {
    // Advance state in its own scope to avoid borrow conflicts
    if !app.pipeline.paused && app.tick.is_multiple_of(60) && app.tick > 0 {
        app.pipeline.active = (app.pipeline.active + 1) % 6;
    }

    let layout = Layout::vertical([
        Constraint::Length(2),  // title
        Constraint::Length(5),  // top row
        Constraint::Length(1),  // connector
        Constraint::Length(5),  // bottom row
        Constraint::Length(1),  // gap
        Constraint::Min(6),     // detail
        Constraint::Length(1),  // hint
    ])
    .margin(1)
    .split(area);

    // Title
    let t = (app.tick as f64 * 0.02).sin() * 0.5 + 0.5;
    f.render_widget(
        Paragraph::new(theme::gradient_line(
            "Pipeline Flow",
            theme::gradient(theme::CYAN, theme::MAGENTA, t),
            theme::gradient(theme::MAGENTA, theme::GOLD, t),
        ))
        .alignment(Alignment::Center),
        layout[0],
    );

    // Top row: stages 0 → 1 → 2
    render_stage_row(f, app, layout[1], [0, 1, 2]);

    // Connector: stage 2 → stage 3
    render_connector(f, app, layout[2]);

    // Bottom row: stages 5 ← 4 ← 3
    render_stage_row(f, app, layout[3], [5, 4, 3]);

    // Detail panel
    render_detail(f, app, layout[5]);

    // Hint
    let paused_indicator = if app.pipeline.paused {
        Span::styled(" ⏸ paused ", Style::default().fg(theme::CORAL))
    } else {
        Span::styled(" ▶ playing ", Style::default().fg(theme::EMERALD))
    };
    f.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(" Space", Style::default().fg(theme::GOLD)),
            Span::styled(" pause  ", Style::default().fg(theme::SLATE)),
            Span::styled("←→", Style::default().fg(theme::GOLD)),
            Span::styled(" step", Style::default().fg(theme::SLATE)),
            Span::styled("  │", Style::default().fg(theme::DIM)),
            paused_indicator,
        ]))
        .alignment(Alignment::Center),
        layout[6],
    );
}

fn render_stage_row(f: &mut Frame, app: &App, area: Rect, indices: [usize; 3]) {
    let cols = Layout::horizontal([
        Constraint::Min(1),
        Constraint::Length(20),
        Constraint::Length(5),
        Constraint::Length(20),
        Constraint::Length(5),
        Constraint::Length(20),
        Constraint::Min(1),
    ])
    .split(area);

    render_stage_box(f, app, cols[1], indices[0]);
    render_arrow(f, app, cols[2], indices[0], indices[1]);
    render_stage_box(f, app, cols[3], indices[1]);
    render_arrow(f, app, cols[4], indices[1], indices[2]);
    render_stage_box(f, app, cols[5], indices[2]);
}

fn render_stage_box(f: &mut Frame, app: &App, area: Rect, idx: usize) {
    let stage = &STAGES[idx];
    let is_active = idx == app.pipeline.active;

    let border_color = if is_active {
        theme::breathe(theme::GOLD, app.tick, 0.08)
    } else {
        theme::DIM
    };

    let border_type = if is_active {
        BorderType::Double
    } else {
        BorderType::Rounded
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(border_type)
        .border_style(Style::default().fg(border_color));

    let icon_color = if is_active {
        theme::gradient(
            theme::CYAN,
            theme::GOLD,
            (app.tick as f64 * 0.05).sin() * 0.5 + 0.5,
        )
    } else {
        theme::SLATE
    };

    let name_style = if is_active {
        theme::bold(theme::TEXT)
    } else {
        Style::default().fg(theme::SLATE)
    };

    let num_color = if is_active { theme::GOLD } else { theme::DIM };

    let content = vec![
        Line::from(Span::styled(
            format!(" {}", idx + 1),
            Style::default().fg(num_color),
        )),
        Line::from(Span::styled(format!(" {}", stage.icon), Style::default().fg(icon_color))),
        Line::from(Span::styled(format!(" {}", stage.name), name_style)),
    ];

    f.render_widget(Paragraph::new(content).block(block), area);
}

fn render_arrow(f: &mut Frame, app: &App, area: Rect, from: usize, to: usize) {
    let active = app.pipeline.active;
    let is_flow = active == from || active == to;
    let color = if is_flow {
        theme::breathe(theme::CYAN, app.tick, 0.1)
    } else {
        theme::DIM
    };

    // Determine arrow direction
    let arrow = if from < to { " ──▸" } else { " ◂──" };

    let y = area.y + area.height / 2;
    f.render_widget(
        Paragraph::new(Span::styled(arrow, Style::default().fg(color))),
        Rect::new(area.x, y, area.width, 1),
    );
}

fn render_connector(f: &mut Frame, app: &App, area: Rect) {
    let active = app.pipeline.active;
    let is_transition = active == 2 || active == 3;
    let color = if is_transition {
        theme::breathe(theme::CYAN, app.tick, 0.1)
    } else {
        theme::DIM
    };

    // Right-aligned vertical connector
    let cols = Layout::horizontal([
        Constraint::Min(1),
        Constraint::Length(20),
        Constraint::Length(5),
        Constraint::Length(20),
        Constraint::Length(5),
        Constraint::Length(20),
        Constraint::Min(1),
    ])
    .split(area);

    // Arrow goes from right box (col 5) down
    let connector_area = cols[5];
    f.render_widget(
        Paragraph::new(Span::styled(
            "          ▾",
            Style::default().fg(color),
        )),
        connector_area,
    );
}

fn render_detail(f: &mut Frame, app: &App, area: Rect) {
    let stage = &STAGES[app.pipeline.active];
    let t = (app.tick as f64 * 0.02).sin() * 0.5 + 0.5;
    let accent = theme::gradient(theme::CYAN, theme::MAGENTA, t);

    let block = Block::default()
        .title(Line::from(vec![
            Span::styled(
                format!(" {} {} ", stage.icon, stage.name),
                theme::bold(accent),
            ),
            Span::styled(
                format!(" {} ", stage.module),
                Style::default().fg(theme::SLATE),
            ),
        ]))
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(accent))
        .padding(Padding::new(2, 2, 1, 1));

    f.render_widget(
        Paragraph::new(Span::styled(stage.detail, Style::default().fg(theme::TEXT)))
            .block(block)
            .wrap(Wrap { trim: true }),
        area,
    );
}
