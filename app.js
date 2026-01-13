// Claude-Style Blog - Minimal JavaScript

(function() {
  const isArticlePage = window.location.pathname.includes('article.html');

  // Initialize Mermaid with Claude-style theme
  if (typeof mermaid !== 'undefined') {
    mermaid.initialize({
      startOnLoad: false,
      theme: 'base',
      themeVariables: {
        primaryColor: '#FAF9F6',
        primaryBorderColor: '#DA7756',
        primaryTextColor: '#2D2D2D',
        lineColor: '#DA7756',
        secondaryColor: '#F5F4F0',
        tertiaryColor: '#FAF9F6'
      }
    });
  }

  // Format date nicely
  function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric'
    });
  }

  // Get slug from URL hash
  function getSlug() {
    return window.location.hash.slice(1);
  }

  // Render math in an element
  function renderMath(element) {
    if (typeof renderMathInElement !== 'undefined') {
      renderMathInElement(element, {
        delimiters: [
          { left: '$$', right: '$$', display: true },
          { left: '$', right: '$', display: false },
          { left: '\\[', right: '\\]', display: true },
          { left: '\\(', right: '\\)', display: false }
        ],
        throwOnError: false
      });
    }
  }

  // Render article list on homepage
  async function renderArticleList() {
    const container = document.getElementById('articles');
    if (!container) return;

    try {
      const response = await fetch('posts/index.json');
      const posts = await response.json();

      if (posts.length === 0) {
        container.innerHTML = '<p>No articles yet.</p>';
        return;
      }

      // Sort by date, newest first
      posts.sort((a, b) => new Date(b.date) - new Date(a.date));

      container.innerHTML = posts.map(post => `
        <a href="article.html#${post.slug}" class="article-link">${post.title}</a>
      `).join('');

      // Render math in titles
      renderMath(container);

    } catch (error) {
      container.innerHTML = '<p>Unable to load articles.</p>';
      console.error('Error loading posts:', error);
    }
  }

  // Render single article
  async function renderArticle() {
    const titleEl = document.getElementById('article-title');
    const dateEl = document.getElementById('article-date');
    const contentEl = document.getElementById('article-content');

    if (!titleEl || !contentEl) return;

    const slug = getSlug();
    if (!slug) {
      window.location.href = 'index.html';
      return;
    }

    try {
      // Get post metadata
      const indexResponse = await fetch('posts/index.json');
      const posts = await indexResponse.json();
      const post = posts.find(p => p.slug === slug);

      if (!post) {
        titleEl.textContent = 'Article not found';
        contentEl.innerHTML = '<p>This article does not exist.</p>';
        return;
      }

      // Update page title and metadata
      document.title = post.title;
      titleEl.innerHTML = post.title;
      dateEl.textContent = formatDate(post.date);

      // Render math in title
      renderMath(titleEl);

      // Load and render markdown
      const mdResponse = await fetch(`posts/${slug}.md`);
      if (!mdResponse.ok) throw new Error('Markdown not found');

      const markdown = await mdResponse.text();
      contentEl.innerHTML = marked.parse(markdown);

      // Syntax highlighting for code blocks
      if (typeof Prism !== 'undefined') {
        Prism.highlightAllUnder(contentEl);
      }

      // Math rendering with KaTeX
      renderMath(contentEl);

      // Render Mermaid diagrams
      if (typeof mermaid !== 'undefined') {
        const mermaidBlocks = contentEl.querySelectorAll('code.language-mermaid');
        mermaidBlocks.forEach((block, index) => {
          const pre = block.parentElement;
          const div = document.createElement('div');
          div.className = 'mermaid';
          div.textContent = block.textContent;
          pre.replaceWith(div);
        });
        mermaid.run();
      }

    } catch (error) {
      contentEl.innerHTML = '<p>Unable to load article.</p>';
      console.error('Error loading article:', error);
    }
  }

  // Initialize
  if (isArticlePage) {
    renderArticle();
    // Re-render when hash changes
    window.addEventListener('hashchange', renderArticle);
  } else {
    renderArticleList();
  }
})();
