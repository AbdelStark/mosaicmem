pub mod dashboard;
pub mod demo;
pub mod explorer;
pub mod pipeline;
pub mod theme;

use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen};
use ratatui::prelude::*;
use ratatui::widgets::*;
use std::io::{self, stdout};
use std::time::Duration;

// ── Tab ──────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Dashboard,
    Pipeline,
    Demo,
    Explorer,
}

impl Tab {
    const ALL: [Tab; 4] = [Tab::Dashboard, Tab::Pipeline, Tab::Demo, Tab::Explorer];

    fn label(&self) -> &'static str {
        match self {
            Tab::Dashboard => " ◈ Dashboard ",
            Tab::Pipeline => " ▷ Pipeline ",
            Tab::Demo => " ◉ Live Demo ",
            Tab::Explorer => " ◫ Architecture ",
        }
    }

    fn index(&self) -> usize {
        match self {
            Tab::Dashboard => 0,
            Tab::Pipeline => 1,
            Tab::Demo => 2,
            Tab::Explorer => 3,
        }
    }
}

// ── App State ────────────────────────────────────────────────

pub struct App {
    pub tab: Tab,
    pub tick: u64,
    pub running: bool,
    pub demo: demo::DemoState,
    pub explorer: explorer::ExplorerState,
    pub pipeline: pipeline::PipelineState,
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

impl App {
    pub fn new() -> Self {
        Self {
            tab: Tab::Dashboard,
            tick: 0,
            running: true,
            demo: demo::DemoState::new(),
            explorer: explorer::ExplorerState::new(),
            pipeline: pipeline::PipelineState::new(),
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
    let tick_rate = Duration::from_millis(33); // ~30 fps

    while app.running {
        terminal.draw(|f| render(f, &mut app))?;

        if event::poll(tick_rate)?
            && let Event::Key(key) = event::read()?
                && key.kind == KeyEventKind::Press {
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
        KeyCode::Tab => {
            let i = (app.tab.index() + 1) % Tab::ALL.len();
            app.tab = Tab::ALL[i];
        }
        KeyCode::BackTab => {
            let i = (app.tab.index() + Tab::ALL.len() - 1) % Tab::ALL.len();
            app.tab = Tab::ALL[i];
        }
        KeyCode::Char('1') => app.tab = Tab::Dashboard,
        KeyCode::Char('2') => app.tab = Tab::Pipeline,
        KeyCode::Char('3') => app.tab = Tab::Demo,
        KeyCode::Char('4') => app.tab = Tab::Explorer,
        other => match app.tab {
            Tab::Demo => demo::handle_key(&mut app.demo, other),
            Tab::Explorer => explorer::handle_key(&mut app.explorer, other),
            Tab::Pipeline => pipeline::handle_key(&mut app.pipeline, other),
            _ => {}
        },
    }
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
        Constraint::Length(3), // tab bar
        Constraint::Min(0),    // content
        Constraint::Length(1), // status bar
    ])
    .split(area);

    render_tab_bar(f, app, layout[0]);

    match app.tab {
        Tab::Dashboard => dashboard::render(f, app, layout[1]),
        Tab::Pipeline => pipeline::render(f, app, layout[1]),
        Tab::Demo => demo::render(f, app, layout[1]),
        Tab::Explorer => explorer::render(f, app, layout[1]),
    }

    render_status_bar(f, layout[2]);
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
        Span::styled(" ◂▸", Style::default().fg(theme::GOLD)),
        Span::styled("/", Style::default().fg(theme::DIM)),
        Span::styled("Tab", Style::default().fg(theme::GOLD)),
        Span::styled(" navigate  ", Style::default().fg(theme::SLATE)),
        Span::styled("1-4", Style::default().fg(theme::GOLD)),
        Span::styled(" jump  ", Style::default().fg(theme::SLATE)),
        Span::styled("q", Style::default().fg(theme::CORAL)),
        Span::styled(" quit", Style::default().fg(theme::SLATE)),
        Span::styled("  │  ", Style::default().fg(theme::DIM)),
        Span::styled("v0.1.0", Style::default().fg(theme::EMERALD)),
        Span::styled("  │  ", Style::default().fg(theme::DIM)),
        Span::styled("arXiv:2603.17117", Style::default().fg(theme::LAVENDER)),
        Span::styled("  │  ", Style::default().fg(theme::DIM)),
        Span::styled("MIT", Style::default().fg(theme::SLATE)),
    ]);

    f.render_widget(
        Paragraph::new(bar).style(Style::default().bg(theme::SURFACE_BRIGHT)),
        area,
    );
}
