pub mod bench_tab;
pub mod coverage_tab;
pub mod demo_tab;
pub mod ops_tab;
pub mod runner;
pub mod theme;

use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::prelude::*;
use ratatui::widgets::*;
use std::io::{self, stdout};
use std::time::{Duration, Instant};

// ── Tab ──────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Demo,
    Coverage,
    Bench,
    Ops,
}

impl Tab {
    const ALL: [Tab; 4] = [Tab::Demo, Tab::Coverage, Tab::Bench, Tab::Ops];

    fn label(&self) -> &'static str {
        match self {
            Tab::Demo => " ▶ Pipeline Demo ",
            Tab::Coverage => " ◧ Coverage ",
            Tab::Bench => " ◈ Benchmark ",
            Tab::Ops => " ⬡ Memory Ops ",
        }
    }

    fn index(&self) -> usize {
        match self {
            Tab::Demo => 0,
            Tab::Coverage => 1,
            Tab::Bench => 2,
            Tab::Ops => 3,
        }
    }
}

// ── App State ────────────────────────────────────────────────

pub struct App {
    pub tab: Tab,
    pub tick: u64,
    pub running: bool,
    pub last_frame: Instant,
    pub transition_tick: Option<u64>,
    pub demo: demo_tab::DemoState,
    pub coverage: coverage_tab::CoverageState,
    pub bench: bench_tab::BenchState,
    pub ops: ops_tab::OpsState,
    tab_initialized: [bool; 4],
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

impl App {
    pub fn new() -> Self {
        Self {
            tab: Tab::Demo,
            tick: 0,
            running: true,
            last_frame: Instant::now(),
            transition_tick: None,
            demo: demo_tab::DemoState::default(),
            coverage: coverage_tab::CoverageState::default(),
            bench: bench_tab::BenchState::default(),
            ops: ops_tab::OpsState::default(),
            tab_initialized: [false; 4],
        }
    }

    fn ensure_tab_init(&mut self) {
        let idx = self.tab.index();
        if self.tab_initialized[idx] {
            return;
        }
        self.tab_initialized[idx] = true;
        match self.tab {
            Tab::Demo => self.demo.init(),
            Tab::Coverage => self.coverage.init(),
            Tab::Bench => self.bench.init(),
            Tab::Ops => self.ops.init(),
        }
    }
}

// ── Entry Point ──────────────────────────────────────────────

pub fn run() -> io::Result<()> {
    enable_raw_mode()?;
    crossterm::execute!(stdout(), EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout()))?;
    terminal.clear()?;

    let mut app = App::new();
    let tick_rate = Duration::from_millis(33);

    // Initialize first tab
    app.ensure_tab_init();

    while app.running {
        terminal.draw(|f| render(f, &mut app))?;

        if event::poll(tick_rate)?
            && let Event::Key(key) = event::read()?
            && key.kind == KeyEventKind::Press
        {
            handle_key(&mut app, key.code);
        }
        app.tick = app.tick.wrapping_add(1);
    }

    disable_raw_mode()?;
    crossterm::execute!(stdout(), LeaveAlternateScreen)?;
    Ok(())
}

// ── Input ────────────────────────────────────────────────────

fn handle_key(app: &mut App, key: KeyCode) {
    match key {
        KeyCode::Char('q') | KeyCode::Esc => app.running = false,
        KeyCode::Tab => switch_tab(app, (app.tab.index() + 1) % Tab::ALL.len()),
        KeyCode::BackTab => {
            switch_tab(app, (app.tab.index() + Tab::ALL.len() - 1) % Tab::ALL.len())
        }
        KeyCode::Char('1') => switch_tab(app, 0),
        KeyCode::Char('2') => switch_tab(app, 1),
        KeyCode::Char('3') => switch_tab(app, 2),
        KeyCode::Char('4') => switch_tab(app, 3),
        KeyCode::Char('r') => {
            // Reset current tab
            let idx = app.tab.index();
            app.tab_initialized[idx] = false;
            match app.tab {
                Tab::Demo => {
                    app.demo = demo_tab::DemoState::default();
                }
                Tab::Coverage => {
                    app.coverage = coverage_tab::CoverageState::default();
                }
                Tab::Bench => {
                    app.bench = bench_tab::BenchState::default();
                }
                Tab::Ops => {
                    app.ops = ops_tab::OpsState::default();
                }
            }
            app.ensure_tab_init();
        }
        _ => {}
    }
}

fn switch_tab(app: &mut App, idx: usize) {
    if idx == app.tab.index() {
        return;
    }
    app.tab = Tab::ALL[idx];
    app.ensure_tab_init();
    // Fire fade-in transition
    app.transition_tick = Some(app.tick);
}

// ── Render ───────────────────────────────────────────────────

fn render(f: &mut Frame, app: &mut App) {
    let area = f.area();

    // Dark background
    f.render_widget(
        Block::default().style(Style::default().bg(theme::SURFACE)),
        area,
    );

    let layout = Layout::vertical([
        Constraint::Length(3),
        Constraint::Min(0),
        Constraint::Length(1),
    ])
    .split(area);

    render_tab_bar(f, app, layout[0]);

    let content_area = layout[1];
    match app.tab {
        Tab::Demo => demo_tab::render(f, &mut app.demo, content_area, app.tick),
        Tab::Coverage => coverage_tab::render(f, &mut app.coverage, content_area, app.tick),
        Tab::Bench => bench_tab::render(f, &mut app.bench, content_area, app.tick),
        Tab::Ops => ops_tab::render(f, &mut app.ops, content_area, app.tick),
    }

    render_status_bar(f, layout[2]);

    // Manual fade-in transition effect
    if let Some(start_tick) = app.transition_tick {
        let age = app.tick.saturating_sub(start_tick);
        let fade_duration = 10u64; // ~330ms at 30fps
        if age < fade_duration {
            let alpha = age as f64 / fade_duration as f64;
            let (sr, sg, sb) = (12u8, 14, 24); // theme::SURFACE rgb
            let buf = f.buffer_mut();
            for y in content_area.top()..content_area.bottom() {
                for x in content_area.left()..content_area.right() {
                    let cell = &mut buf[(x, y)];
                    if let Color::Rgb(r, g, b) = cell.fg {
                        cell.fg = Color::Rgb(
                            (sr as f64 + (r as f64 - sr as f64) * alpha) as u8,
                            (sg as f64 + (g as f64 - sg as f64) * alpha) as u8,
                            (sb as f64 + (b as f64 - sb as f64) * alpha) as u8,
                        );
                    }
                }
            }
        } else {
            app.transition_tick = None;
        }
    }
}

fn render_tab_bar(f: &mut Frame, app: &App, area: Rect) {
    let titles: Vec<&str> = Tab::ALL.iter().map(|t| t.label()).collect();

    let block = Block::default()
        .borders(Borders::BOTTOM)
        .border_style(Style::default().fg(theme::DIM))
        .title(
            Line::from(vec![
                Span::styled(" M", theme::bold(theme::CYAN)),
                Span::styled("osaic", Style::default().fg(theme::TEXT)),
                Span::styled("M", theme::bold(theme::MAGENTA)),
                Span::styled("em ", Style::default().fg(theme::TEXT)),
            ])
            .alignment(Alignment::Center),
        );

    let tabs = Tabs::new(titles)
        .block(block)
        .select(app.tab.index())
        .style(Style::default().fg(theme::SLATE))
        .highlight_style(
            Style::default()
                .fg(theme::GOLD)
                .add_modifier(Modifier::BOLD),
        )
        .divider(Span::styled(" │ ", Style::default().fg(theme::DIM)));

    f.render_widget(tabs, area);
}

fn render_status_bar(f: &mut Frame, area: Rect) {
    let bar = Line::from(vec![
        Span::styled(" Tab", Style::default().fg(theme::GOLD)),
        Span::styled(" navigate  ", Style::default().fg(theme::SLATE)),
        Span::styled("1-4", Style::default().fg(theme::GOLD)),
        Span::styled(" jump  ", Style::default().fg(theme::SLATE)),
        Span::styled("r", Style::default().fg(theme::GOLD)),
        Span::styled(" rerun  ", Style::default().fg(theme::SLATE)),
        Span::styled("q", Style::default().fg(theme::CORAL)),
        Span::styled(" quit", Style::default().fg(theme::SLATE)),
        Span::styled("  │  ", Style::default().fg(theme::DIM)),
        Span::styled("v0.1.0", Style::default().fg(theme::EMERALD)),
        Span::styled("  │  ", Style::default().fg(theme::DIM)),
        Span::styled("crates.io/mosaicmem", Style::default().fg(theme::LAVENDER)),
        Span::styled("  │  ", Style::default().fg(theme::DIM)),
        Span::styled("arXiv:2603.17117", Style::default().fg(theme::SLATE)),
    ]);

    f.render_widget(
        Paragraph::new(bar).style(Style::default().bg(theme::SURFACE_1)),
        area,
    );
}
