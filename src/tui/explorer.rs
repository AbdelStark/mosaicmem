use crossterm::event::KeyCode;
use ratatui::prelude::*;
use ratatui::widgets::*;

use super::theme;
use super::App;

pub struct ExplorerState {
    pub selected: usize,
}

impl Default for ExplorerState {
    fn default() -> Self {
        Self::new()
    }
}

impl ExplorerState {
    pub fn new() -> Self {
        Self { selected: 0 }
    }
}

pub fn handle_key(state: &mut ExplorerState, key: KeyCode) {
    match key {
        KeyCode::Down | KeyCode::Char('j') => state.selected = (state.selected + 1) % MODULES.len(),
        KeyCode::Up | KeyCode::Char('k') => {
            state.selected = (state.selected + MODULES.len() - 1) % MODULES.len()
        }
        _ => {}
    }
}

struct Module {
    name: &'static str,
    icon: &'static str,
    files: usize,
    description: &'static str,
    types: &'static [(&'static str, &'static str)],
}

const MODULES: [Module; 6] = [
    Module {
        name: "attention",
        icon: "◈",
        files: 6,
        description: "Position encoding and cross-attention mechanisms for spatial memory \
                       conditioning. Implements Warped RoPE for geometry-aware attention, \
                       Warped Latent for feature-level alignment, and PRoPE for progressive \
                       temporal encoding with decay.",
        types: &[
            ("WarpedRoPE", "Geometry-aware rotary position encoding"),
            ("WarpedLatent", "Feature-level view reprojection"),
            ("PRoPE", "Progressive rotary PE with temporal decay"),
            ("MemoryCrossAttention", "Memory injection into diffusion backbone"),
        ],
    },
    Module {
        name: "camera",
        icon: "◎",
        files: 4,
        description: "Camera model, pose representation, and trajectory generation. \
                       Supports arbitrary 6-DoF camera paths with SE(3) pose composition, \
                       JSON serialization, and built-in trajectory generators (circle, \
                       figure-eight, linear).",
        types: &[
            ("CameraPose", "6-DoF camera pose (position + quaternion)"),
            ("CameraIntrinsics", "Focal length, principal point, resolution"),
            ("CameraTrajectory", "Ordered sequence of timestamped poses"),
        ],
    },
    Module {
        name: "diffusion",
        icon: "▣",
        files: 4,
        description: "Video generation backbone with DDPM noise scheduling and VAE \
                       encode/decode. Ships deterministic synthetic implementations \
                       that produce structured output for end-to-end testing without \
                       GPU or model weights.",
        types: &[
            ("SyntheticBackbone", "Deterministic diffusion backbone for testing"),
            ("DDPMScheduler", "Denoising diffusion probabilistic scheduling"),
            ("SyntheticVAE", "Latent encode/decode without neural networks"),
        ],
    },
    Module {
        name: "geometry",
        icon: "△",
        files: 5,
        description: "3D reconstruction primitives: monocular depth estimation, \
                       point cloud construction, streaming voxel fusion, and camera \
                       projection/unprojection. Builds the spatial foundation that \
                       the memory system indexes over.",
        types: &[
            ("SyntheticDepthEstimator", "Structured depth map generation"),
            ("PointCloud3D", "Colored point cloud with normals"),
            ("StreamingFusion", "Incremental voxel-grid point cloud fusion"),
        ],
    },
    Module {
        name: "memory",
        icon: "▦",
        files: 5,
        description: "Spatial memory store backed by a kd-tree (kiddo) for O(log n) \
                       nearest-neighbor retrieval. Handles patch insertion, frustum-aware \
                       retrieval with temporal decay, diversity filtering, coverage \
                       tracking, and scene manipulation (splice, transform, erase).",
        types: &[
            ("MosaicMemoryStore", "Spatial patch store with kd-tree index"),
            ("Patch3D", "3D-positioned image patch with latent features"),
            ("MosaicFrame", "Retrieved patches + coverage mask for a view"),
            ("MemoryRetriever", "View-conditioned patch retrieval engine"),
        ],
    },
    Module {
        name: "pipeline",
        icon: "▷",
        files: 4,
        description: "End-to-end video generation pipeline with autoregressive \
                       windowing for arbitrary-length output. Orchestrates depth \
                       estimation, memory updates, retrieval, alignment, and diffusion \
                       denoising across overlapping temporal windows.",
        types: &[
            ("AutoregressivePipeline", "Multi-window generation with memory carryover"),
            ("InferencePipeline", "Single-window inference with full conditioning"),
            ("PipelineConfig", "19-field configuration for all pipeline parameters"),
            ("PipelineStats", "Live metrics: patches, points, keyframes, tokens"),
        ],
    },
];

pub fn render(f: &mut Frame, app: &mut App, area: Rect) {
    let cols = Layout::horizontal([Constraint::Length(26), Constraint::Min(0)])
        .margin(1)
        .split(area);

    render_module_list(f, app, cols[0]);
    render_module_detail(f, app, cols[1]);
}

fn render_module_list(f: &mut Frame, app: &App, area: Rect) {
    let selected = app.explorer.selected;

    let items: Vec<ListItem> = MODULES
        .iter()
        .enumerate()
        .map(|(i, m)| {
            let is_sel = i == selected;
            let t = i as f64 / (MODULES.len() - 1) as f64;
            let icon_color = if is_sel {
                theme::GOLD
            } else {
                theme::gradient(theme::CYAN, theme::MAGENTA, t)
            };

            let name_style = if is_sel {
                theme::bold(theme::TEXT)
            } else {
                Style::default().fg(theme::TEXT_MUTED)
            };

            let indicator = if is_sel { "▸" } else { " " };

            ListItem::new(Line::from(vec![
                Span::styled(format!(" {indicator} "), Style::default().fg(theme::GOLD)),
                Span::styled(format!("{} ", m.icon), Style::default().fg(icon_color)),
                Span::styled(m.name, name_style),
                Span::styled(
                    format!("  {}f", m.files),
                    Style::default().fg(theme::DIM),
                ),
            ]))
        })
        .collect();

    let block = Block::default()
        .title(Span::styled(" Modules ", theme::title_style()))
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(theme::border_style())
        .padding(Padding::vertical(1));

    f.render_widget(List::new(items).block(block), area);
}

fn render_module_detail(f: &mut Frame, app: &App, area: Rect) {
    let m = &MODULES[app.explorer.selected];
    let t = app.explorer.selected as f64 / (MODULES.len() - 1) as f64;
    let accent = theme::gradient(theme::CYAN, theme::MAGENTA, t);

    let rows = Layout::vertical([
        Constraint::Length(6),  // description
        Constraint::Min(0),     // types table
    ])
    .split(area);

    // Description
    let desc_block = Block::default()
        .title(Line::from(vec![
            Span::styled(format!(" {} {} ", m.icon, m.name), theme::bold(accent)),
            Span::styled(
                format!(" {} files ", m.files),
                Style::default().fg(theme::SLATE),
            ),
        ]))
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(accent))
        .padding(Padding::new(2, 2, 1, 0));

    f.render_widget(
        Paragraph::new(Span::styled(m.description, Style::default().fg(theme::TEXT)))
            .block(desc_block)
            .wrap(Wrap { trim: true }),
        rows[0],
    );

    // Types table
    let type_rows: Vec<Row> = m
        .types
        .iter()
        .enumerate()
        .map(|(i, (name, desc))| {
            let tt = i as f64 / m.types.len().max(1) as f64;
            let type_color = theme::gradient(accent, theme::GOLD, tt);
            Row::new(vec![
                Cell::from(Span::styled(format!("  {name}"), theme::bold(type_color))),
                Cell::from(Span::styled(*desc, Style::default().fg(theme::TEXT_MUTED))),
            ])
        })
        .collect();

    let types_block = Block::default()
        .title(Span::styled(" Key Types ", theme::title_style()))
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(theme::border_style())
        .padding(Padding::vertical(1));

    let table = Table::new(
        type_rows,
        [Constraint::Length(24), Constraint::Min(0)],
    )
    .block(types_block);

    f.render_widget(table, rows[1]);
}
