import { useState, useRef, useEffect, useMemo } from 'react'
import { askQuestion, expandContext, API_URL } from './api'
import './App.css'
import jsPDF from 'jspdf'

// Chat type: { id: string, title: string, messages: Message[], createdAt: number, summary: string, lastTitleUpdateAt: number, userMessageCountAtLastUpdate: number, isTitleManual: boolean }

const STORAGE_KEY_CHATS = 'rag_pdf_chats'
const STORAGE_KEY_ACTIVE_CHAT = 'rag_pdf_active_chat_id'
const STORAGE_KEY_THEME = 'rag_pdf_theme'

function App() {
  const [chats, setChats] = useState([])
  const [activeChatId, setActiveChatId] = useState(null)
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [typing] = useState(false)
  const messagesEndRef = useRef(null)
  const [editingChatId, setEditingChatId] = useState(null)
  const [editingTitle, setEditingTitle] = useState('')
  const [searchQuery, setSearchQuery] = useState('')
  const [showContext, setShowContext] = useState(false)
  const [theme, setTheme] = useState(() => {
    try {
      const savedTheme = localStorage.getItem(STORAGE_KEY_THEME)
      return savedTheme === 'dark' ? 'dark' : 'light'
    } catch {
      return 'light'
    }
  })

  // Apply theme on mount
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
  }, [theme])

  // Load chats and activeChatId from localStorage on mount
  useEffect(() => {
    try {
      const savedChats = localStorage.getItem(STORAGE_KEY_CHATS)
      const savedActiveChatId = localStorage.getItem(STORAGE_KEY_ACTIVE_CHAT)
      
      if (savedChats) {
        const parsedChats = JSON.parse(savedChats)
        const normalizedChats = parsedChats.map(chat => ({
          ...chat,
          summary: chat.summary || '',
          lastTitleUpdateAt: chat.lastTitleUpdateAt || 0,
          userMessageCountAtLastUpdate: chat.userMessageCountAtLastUpdate || 0,
          isTitleManual: chat.isTitleManual || false
        }))
        
        setChats(normalizedChats)
        
        if (savedActiveChatId && normalizedChats.find(c => c.id === savedActiveChatId)) {
          setActiveChatId(savedActiveChatId)
        } else if (normalizedChats.length > 0) {
          setActiveChatId(normalizedChats[0].id)
        } else {
          setActiveChatId(null)
        }
      } else {
        setChats([])
        setActiveChatId(null)
      }
    } catch (error) {
      console.error('Error loading chats from localStorage:', error)
      setChats([])
      setActiveChatId(null)
    }
  }, [])

  // Save chats and activeChatId to localStorage on every change
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY_CHATS, JSON.stringify(chats))
      localStorage.setItem(STORAGE_KEY_ACTIVE_CHAT, activeChatId || '')
    } catch (error) {
      console.error('Error saving chats to localStorage:', error)
    }
  }, [chats, activeChatId])

  // Save theme to localStorage and apply to document
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY_THEME, theme)
      document.documentElement.setAttribute('data-theme', theme)
    } catch (error) {
      console.error('Error saving theme to localStorage:', error)
    }
  }, [theme])

  const activeChat = useMemo(() => 
    chats.find(chat => chat.id === activeChatId),
    [chats, activeChatId]
  )
  const activeMessages = useMemo(() => activeChat?.messages || [], [activeChat])

  // Auto-create chat if no chats exist
  useEffect(() => {
    if (chats.length === 0) {
      const defaultChat = {
        id: `chat-${Date.now()}`,
        title: 'New Chat',
        messages: [],
        createdAt: Date.now(),
        summary: '',
        lastTitleUpdateAt: 0,
        userMessageCountAtLastUpdate: 0,
        isTitleManual: false
      }
      setChats([defaultChat])
      setActiveChatId(defaultChat.id)
    } else if (!activeChatId && chats.length > 0) {
      // If no active chat but chats exist, select the most recent
      const sorted = [...chats].sort((a, b) => b.createdAt - a.createdAt)
      setActiveChatId(sorted[0].id)
    }
  }, [chats, activeChatId])

  // Filter chats based on search query
  const filteredChats = chats.filter(chat => {
    if (!searchQuery.trim()) return true

    const q = searchQuery.toLowerCase()

    // Search title
    if (chat.title?.toLowerCase().includes(q)) return true

    // Search messages
    return chat.messages.some(msg => {
      if (msg.type === 'user' && msg.text?.toLowerCase().includes(q)) {
        return true
      }
      if (msg.type === 'ai') {
        const text = getPlainTextFromBlocks(msg.blocks)
        return text.toLowerCase().includes(q)
      }
      return false
    })
  })

  // Create new chat
  const handleNewChat = () => {
    const newChat = {
      id: `chat-${Date.now()}`,
      title: 'New Chat',
      messages: [],
      createdAt: Date.now(),
      summary: '',
      lastTitleUpdateAt: 0,
      userMessageCountAtLastUpdate: 0,
      isTitleManual: false
    }
    setChats(prev => [...prev, newChat])
    setActiveChatId(newChat.id)
    setInput('')
  }

  // Handle rename chat
  const handleRenameStart = (chatId, currentTitle) => {
    setEditingChatId(chatId)
    setEditingTitle(currentTitle)
  }

  const handleRenameConfirm = (chatId) => {
    if (editingTitle.trim()) {
      setChats(prev => prev.map(chat => 
        chat.id === chatId
          ? { ...chat, title: editingTitle.trim(), isTitleManual: true }
          : chat
      ))
    }
    setEditingChatId(null)
    setEditingTitle('')
  }

  const handleRenameCancel = () => {
    setEditingChatId(null)
    setEditingTitle('')
  }

  // Handle delete chat
  const handleDeleteChat = (chatId) => {
    if (!window.confirm('Are you sure you want to delete this chat?')) {
      return
    }

    const updated = chats.filter(chat => chat.id !== chatId)
    
    // If deleted chat was active, switch to most recent or create new
    if (chatId === activeChatId) {
      if (updated.length > 0) {
        // Sort by createdAt and select most recent
        const sorted = [...updated].sort((a, b) => b.createdAt - a.createdAt)
        setActiveChatId(sorted[0].id)
      } else {
        // Create new chat if none remain
        const newChat = {
          id: `chat-${Date.now()}`,
          title: 'New Chat',
          messages: [],
          createdAt: Date.now(),
          summary: '',
          lastTitleUpdateAt: 0,
          userMessageCountAtLastUpdate: 0,
          isTitleManual: false
        }
        setActiveChatId(newChat.id)
        setChats([newChat])
        return
      }
    }
    
    setChats(updated)
  }

  // Switch to a different chat
  const handleChatClick = (chatId) => {
    setActiveChatId(chatId)
    setInput('')
  }

  // Generate chat title using Gemini (called exactly once when first AI message is added)
  const generateChatTitle = async (chatId, question) => {
    // Set loading placeholder
    setChats(prev => prev.map(chat => 
      chat.id === chatId && chat.title === 'New Chat' && !chat.isTitleManual
        ? { ...chat, title: 'Thinking…' }
        : chat
    ))

    try {
      // Call Gemini API directly
      const API_KEY = import.meta.env.VITE_GEMINI_API_KEY
      if (!API_KEY) {
        throw new Error('GEMINI_API_KEY not configured')
      }

      const prompt = `Generate a concise chat title (3–6 words) summarizing the topic.
Rules:
- No punctuation
- No markdown
- No filler words like explain, describe
- Output ONLY the title text

User question:
${question}`

      const response = await fetch(
        `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=${API_KEY}`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            contents: [{
              parts: [{
                text: prompt
              }]
            }]
          })
        }
      )

      if (!response.ok) {
        throw new Error('Failed to generate title')
      }

      const data = await response.json()
      const title = data.candidates?.[0]?.content?.parts?.[0]?.text?.trim() || 'New Chat'
      
      // Clean up title (remove any extra formatting)
      const cleanTitle = title.replace(/[^\w\s]/g, '').trim() || 'New Chat'
      
      // Update chat title only if not manually set
      setChats(prev => {
        const chat = prev.find(c => c.id === chatId)
        if (chat?.isTitleManual) {
          return prev // Don't override manual titles
        }
        return prev.map(c => 
          c.id === chatId
            ? { ...c, title: cleanTitle }
            : c
        )
      })
    } catch (error) {
      console.error('Error generating title:', error)
      // Fallback to 'New Chat' on error
      setChats(prev => prev.map(chat => 
        chat.id === chatId && chat.title === 'Thinking…' && !chat.isTitleManual
          ? { ...chat, title: 'New Chat' }
          : chat
      ))
    }
  }

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [activeMessages])

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!input.trim() || loading || typing || !activeChatId) return

    const userMessage = { type: 'user', text: input }
    const questionText = input
    
    // Calculate updated summary
    const currentSummary = activeChat?.summary || ''
    const newSummary = (currentSummary + ' ' + questionText).trim()
    let updatedSummary = newSummary
    if (newSummary.length > 400) {
      updatedSummary = newSummary.substring(0, 400)
      const lastSpace = updatedSummary.lastIndexOf(' ')
      if (lastSpace > 0) {
        updatedSummary = updatedSummary.substring(0, lastSpace)
      }
    }
    
    // Add user message and update summary in one state update
    setChats(prev => prev.map(chat => 
      chat.id === activeChatId 
        ? { ...chat, messages: [...chat.messages, userMessage], summary: updatedSummary }
        : chat
    ))
    
    setInput('')
    setLoading(true)

    try {
      // PDF Q/A ONLY: Call /ask endpoint
      const response = await askQuestion(questionText)
      
      // STRICT VALIDATION: Ensure response has expected PDF Q/A structure
      if (!response || typeof response !== 'object') {
        throw new Error('Invalid response format from server')
      }

      // Backend may return a structured error (e.g., exact reference not found)
      if (response.error) {
        const errorMessage = {
          type: 'ai',
          blocks: [
            { type: 'paragraph', text: response.error }
          ],
          sources: [],
          intent: response.intent,
          target_id: response.target_id,
        }

        setChats(prev => prev.map(chat => 
          chat.id === activeChatId 
            ? { ...chat, messages: [...chat.messages, errorMessage] }
            : chat
        ))
        setLoading(false)
        return
      }
      
      // PDF Q/A ONLY: Extract blocks and sources (never drawing analysis)
      const blocks = Array.isArray(response.blocks) ? response.blocks : []
      const sources = Array.isArray(response.sources) ? response.sources : []
      
      // FAILSAFE: If blocks are empty, show error
      if (blocks.length === 0) {
        throw new Error('No answer blocks received from server')
      }
      
      const aiMessage = {
        type: 'ai',
        blocks: blocks,
        sources: sources,
        isTyping: true,
        intent: response.intent,
        target_id: response.target_id,
      }
      
      // Add assistant message to active chat
      setChats(prev => prev.map(chat => {
        if (chat.id !== activeChatId) return chat
        
        const updatedMessages = [...chat.messages, aiMessage]
        
        // Auto-generate title exactly once when first AI message is added
        if (updatedMessages.filter(m => m.type === 'ai').length === 1) {
          if (chat.title === 'New Chat' && !chat.isTitleManual) {
            generateChatTitle(chat.id, questionText)
          }
        }
        
        return {
          ...chat,
          messages: updatedMessages
        }
      }))
      setLoading(false)
    } catch (error) {
      console.error('PDF Q/A Error:', error)
      
      // FAILSAFE: Show readable error message
      const errorMessage = {
        type: 'ai',
        blocks: [
          {
            type: 'paragraph',
            text: error.message || 'Sorry, I encountered an error. Please try again.'
          }
        ],
        sources: []
      }
      
      // Add error message to active chat (preserve previous messages)
      setChats(prev => prev.map(chat => {
        if (chat.id !== activeChatId) return chat
        
        const updatedMessages = [...chat.messages, errorMessage]
        
        // Auto-generate title exactly once when first AI message (including errors) is added
        if (updatedMessages.filter(m => m.type === 'ai').length === 1) {
          if (chat.title === 'New Chat' && !chat.isTitleManual) {
            generateChatTitle(chat.id, questionText)
          }
        }
        
        return {
          ...chat,
          messages: updatedMessages
        }
      }))
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <div className="app-layout">
        <div className="sidebar">
          <button className="new-chat-button" onClick={handleNewChat}>
            New Chat
          </button>
          <div className="chat-list-title">
            PDF Chats
          </div>
          <input
            type="text"
            className="chat-search-input"
            placeholder="Search chats"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
          <div className="chat-list">
            {filteredChats.map(chat => (
              <div
                key={chat.id}
                className={`chat-item ${chat.id === activeChatId ? 'active' : ''}`}
              >
                {editingChatId === chat.id ? (
                  <input
                    type="text"
                    value={editingTitle}
                    onChange={(e) => setEditingTitle(e.target.value)}
                    onBlur={() => handleRenameConfirm(chat.id)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') {
                        handleRenameConfirm(chat.id)
                      } else if (e.key === 'Escape') {
                        handleRenameCancel()
                      }
                    }}
                    onClick={(e) => e.stopPropagation()}
                    className="chat-title-input"
                    autoFocus
                  />
                ) : (
                  <>
                    <span
                      className="chat-title-text"
                      onClick={() => handleChatClick(chat.id)}
                    >
                      {chat.title}
                    </span>
                    <div className="chat-actions">
                      <button
                        className="chat-action-button"
                        onClick={(e) => {
                          e.stopPropagation()
                          handleRenameStart(chat.id, chat.title)
                        }}
                        title="Rename"
                      >
                        ✏️
                      </button>
                      <button
                        className="chat-action-button"
                        onClick={(e) => {
                          e.stopPropagation()
                          handleDeleteChat(chat.id)
                        }}
                        title="Delete"
                      >
                        🗑️
                      </button>
                    </div>
                  </>
                )}
              </div>
            ))}
          </div>
        </div>
        <div className="chat-container">
          <div className="header-bar">
            <button
              className="theme-toggle"
              onClick={() => setTheme(theme === 'light' ? 'dark' : 'light')}
              title={theme === 'light' ? 'Switch to dark mode' : 'Switch to light mode'}
            >
              {theme === 'light' ? '🌙' : '☀️'}
            </button>
          </div>
          {activeChat && activeChat.messages.length > 0 && (
            <div className="chat-header">
              <button 
                className="context-button"
                onClick={() => setShowContext(true)}
                title="View context window"
              >
                Context
              </button>
              <button 
                className="export-pdf-button"
                onClick={() => exportChatToPDF(activeChat)}
                title="Export chat to PDF"
              >
                📄 Export PDF
              </button>
            </div>
          )}
          <div className="messages">
            {activeMessages.length === 0 && (
              <div className="welcome-message">
                <h2>RAG PDF Assistant</h2>
                <p>Ask questions about the uploaded document</p>
              </div>
            )}
            {activeMessages.map((msg, idx) => {
              if (msg.type === 'ai') {
                return (
                  <div key={idx} className={`message ${msg.type}`}>
                    <div className="message-content">
                      <AssistantMessage blocks={msg.blocks || []} intent={msg.intent} targetId={msg.target_id} />
                    </div>
                  </div>
                )
              }
              // User messages
              return (
                <div key={idx} className={`message ${msg.type}`}>
                  <div className="message-content">
                    {msg.text}
                  </div>
                </div>
              )
            })}
            {loading && (
              <div className="message ai">
                <div className="message-content typing-indicator">
                  <span className="typing-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                  </span>
                  <span className="typing-text">Assistant is typing…</span>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
          <form className="input-form" onSubmit={handleSubmit}>
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask a question..."
              disabled={loading || typing}
            />
            <button type="submit" disabled={loading || typing || !input.trim()}>
              Send
            </button>
          </form>
        </div>
      </div>
      {showContext && activeChat && (
        <div className="context-overlay" onClick={() => setShowContext(false)}>
          <div className="context-panel" onClick={(e) => e.stopPropagation()}>
            <div className="context-header">
              <h3>Context Window</h3>
              <button 
                className="context-close-button"
                onClick={() => setShowContext(false)}
              >
                Close
              </button>
            </div>
            <div className="context-content">
              <pre>{buildContextPreview(activeChat.messages)}</pre>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function getPlainTextFromBlocks(blocks) {
  if (!blocks || blocks.length === 0) return ''

  return blocks.map(block => {
    if (block.type === 'paragraph') {
      return block.text
    }
    if (block.type === 'list' && Array.isArray(block.items)) {
      return block.items.join('\n')
    }
    return ''
  }).join('\n\n')
}

function buildContextPreview(messages, maxPairs = 3) {
  if (!messages || messages.length === 0) return ''

  const recent = messages.slice(-maxPairs * 2)

  return recent.map(msg => {
    if (msg.type === 'user') {
      return `User: ${msg.text}`
    }
    if (msg.type === 'ai') {
      const text = getPlainTextFromBlocks(msg.blocks)
      return `Assistant: ${text}`
    }
    return ''
  }).join('\n\n')
}

function exportChatToPDF(chat) {
  if (!chat) return

  const doc = new jsPDF()
  let y = 20
  const pageHeight = doc.internal.pageSize.height
  const margin = 10
  const maxWidth = 190

  // Title
  doc.setFontSize(16)
  doc.setFont('helvetica', 'bold')
  const titleLines = doc.splitTextToSize(chat.title || 'Chat Export', maxWidth)
  doc.text(titleLines, margin, y)
  y += titleLines.length * 7 + 10

  doc.setFontSize(11)
  doc.setFont('helvetica', 'normal')

    chat.messages.forEach(msg => {
      // Check if we need a new page
      if (y > pageHeight - 30) {
        doc.addPage()
        y = 20
      }

      if (msg.type === 'user') {
        doc.setFont('helvetica', 'bold')
        doc.text('User:', margin, y)
        y += 6

        doc.setFont('helvetica', 'normal')
        const userLines = doc.splitTextToSize(msg.text || '', maxWidth)
        doc.text(userLines, margin, y)
        y += userLines.length * 6 + 10
      }

      if (msg.type === 'ai') {
        doc.setFont('helvetica', 'bold')
        doc.text('Assistant:', margin, y)
        y += 6

        doc.setFont('helvetica', 'normal')

        const text = getPlainTextFromBlocks(msg.blocks)
        const lines = doc.splitTextToSize(text, maxWidth)
        doc.text(lines, margin, y)
        y += lines.length * 6 + 10
      }

    })

  const sanitizedTitle = (chat.title || 'chat').replace(/[<>:"/\\|?*]/g, '_').trim()
  doc.save(`${sanitizedTitle || 'chat'}.pdf`)
}

function highlightText(text, phrases) {
  if (!phrases || phrases.length === 0 || !text) {
    return text
  }

  const textLower = text.toLowerCase()
  const matches = []

  phrases.forEach(phrase => {
    if (!phrase || phrase.trim().length === 0) return
    
    const phraseLower = phrase.toLowerCase().trim()
    const index = textLower.indexOf(phraseLower)
    
    if (index !== -1) {
      const endIndex = index + phrase.length
      let overlap = false
      
      for (const match of matches) {
        if (!(endIndex <= match.start || index >= match.end)) {
          overlap = true
          break
        }
      }
      
      if (!overlap) {
        matches.push({ start: index, end: endIndex, phrase })
      }
    }
  })

  if (matches.length === 0) {
    return text
  }

  matches.sort((a, b) => a.start - b.start)

  let result = ''
  let lastIndex = 0

  matches.forEach(match => {
    if (match.start > lastIndex) {
      result += text.substring(lastIndex, match.start)
    }
    const originalText = text.substring(match.start, match.end)
    result += '<mark class="highlight-phrase">' + originalText + '</mark>'
    lastIndex = match.end
  })

  if (lastIndex < text.length) {
    result += text.substring(lastIndex)
  }

  return result || text
}

function sortSourcesByIntent(sources, intent) {
  if (!Array.isArray(sources)) return []
  const list = [...sources]

  if (intent === 'FIGURE_QUERY') {
    // Image must appear above text.
    const pri = (s) => (s?.source === 'IMAGE' ? 0 : 1)
    return list.sort((a, b) => pri(a) - pri(b))
  }

  if (intent === 'TABLE_QUERY') {
    // Table must appear first.
    const pri = (s) => (s?.source === 'TABLE' ? 0 : 1)
    return list.sort((a, b) => pri(a) - pri(b))
  }

  return list
}

function SourceItem({ source, isPrimary = false, intent, targetId }) {
  const [expanded, setExpanded] = useState(false)
  const [showOcrDetails, setShowOcrDetails] = useState(false)

  // Intent-aware strict rendering rules
  const isFigureQuery = intent === 'FIGURE_QUERY'
  const isTableQuery = intent === 'TABLE_QUERY'
  const isStrict = isFigureQuery || isTableQuery

  const effectiveExpanded = isStrict ? true : expanded

  const isImage = source?.source === 'IMAGE'
  const isTable = source?.source === 'TABLE'
  const isOcr = source?.source === 'OCR'

  const renderHighlightedText = (text, phrases) => {
    if (isStrict || !phrases || phrases.length === 0) {
      return text
    }
    
    const highlighted = highlightText(text, phrases)
    return <span dangerouslySetInnerHTML={{ __html: highlighted }} />
  }

  const getFigureCaptionLine = () => {
    if (!isImage) return null
    const figId = (targetId || '').trim()
    const captionText = (source.caption || '').trim()
    if (figId && captionText) return `${figId} — ${captionText}`
    if (figId) return figId
    return captionText || null
  }

  const parseTable = (rawText) => {
    const text = (rawText || '').trim()
    if (!text) return { rows: [] }

    let lines = text.split('\n').map(l => l.trimEnd()).filter(Boolean)

    // Drop a leading "Table X.X ..." title line if it exists, keep the rest for rows.
    if (lines.length > 1 && /^table\s*\d+(\.\d+)*\b/i.test(lines[0])) {
      lines = lines.slice(1)
    }

    const hasPipes = lines.some(l => l.includes('|'))
    const hasTabs = lines.some(l => /\t+/.test(l))
    const hasMultiSpace = lines.some(l => /\s{2,}/.test(l))

    let splitter = null
    if (hasPipes) splitter = (line) => line.split('|').map(c => c.trim())
    else if (hasTabs) splitter = (line) => line.split(/\t+/).map(c => c.trim())
    else if (hasMultiSpace) splitter = (line) => line.split(/\s{2,}/).map(c => c.trim())
    else splitter = (line) => [line]

    const rows = lines.map(splitter).filter(r => r.length > 0)
    const maxCols = rows.reduce((m, r) => Math.max(m, r.length), 0)
    const normalized = rows.map(r => {
      const out = [...r]
      while (out.length < maxCols) out.push('')
      return out
    })

    return { rows: normalized }
  }

  return (
    <div className="source-item">
      <div
        className={`source-header ${isStrict ? 'source-header-static' : ''}`}
        onClick={() => {
          if (isStrict) return
          setExpanded(!expanded)
        }}
      >
        <div className="source-header-left">
          {isPrimary && (
            <div className="primary-source-badge">PRIMARY SOURCE</div>
          )}
          <span className="source-info">
            Page {source.page_number || 'N/A'} • {source.source}
            {isOcr && <span className="source-badge ocr-badge">OCR extracted</span>}
          </span>
        </div>
        {!isStrict && <span className="expand-toggle">{expanded ? '−' : '+'}</span>}
      </div>
      {effectiveExpanded && (
        <div className="source-details">
          {/* TABLE_QUERY: table must appear first */}
          {isTable && (
            <div className="source-table-container">
              {(() => {
                const { rows } = parseTable(source.raw_text || source.text || '')
                if (!rows || rows.length === 0) return null
                return (
                  <table className="source-table">
                    <tbody>
                      {rows.map((row, rIdx) => (
                        <tr key={rIdx}>
                          {row.map((cell, cIdx) => (
                            <td key={cIdx}>{cell}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )
              })()}
            </div>
          )}

          {/* IMAGE sources */}
          {isImage && source.image_path && (
            <div className="source-media">
              <a
                href={`${API_URL}/${source.image_path}`}
                target="_blank"
                rel="noopener noreferrer"
                className="source-media-link"
                title="Open full page"
              >
                <img
                  className="source-thumb"
                  src={`${API_URL}/${source.image_path}`}
                  alt="Source thumbnail"
                  loading="lazy"
                />
              </a>
              {getFigureCaptionLine() && (
                <div className="source-caption">{getFigureCaptionLine()}</div>
              )}
            </div>
          )}

          {/* Excerpt display rules:
              - Strict (FIGURE/TABLE): always show excerpt (no expand)
              - Otherwise: show expandable excerpt
          */}
          {isStrict && source.text && (
            <div className="source-excerpt">
              <pre className="source-pre">{source.text}</pre>
            </div>
          )}

          {!isStrict && source.text && (
            <div className="ocr-text-compact">
              <button
                type="button"
                className="show-details-btn"
                onClick={(e) => {
                  e.preventDefault()
                  e.stopPropagation()
                  setShowOcrDetails(!showOcrDetails)
                }}
              >
                {showOcrDetails ? 'Hide excerpt' : 'Show excerpt'}
              </button>
              <div className={`source-text-expandable ${showOcrDetails ? 'expanded' : 'collapsed'}`}>
                <div className="source-text">
                  {renderHighlightedText(source.text, source.highlight_phrases)}
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function AssistantMessage({ blocks = [], intent, targetId }) {
  const [copied, setCopied] = useState(false)
  const [contextByBlockIdx, setContextByBlockIdx] = useState({})
  const [contextUiByBlockIdx, setContextUiByBlockIdx] = useState({})

  if (!blocks || blocks.length === 0) {
    return <p>⚠️ No answer returned.</p>;
  }

  const handleCopy = async () => {
    const text = getPlainTextFromBlocks(blocks)
    try {
      await navigator.clipboard.writeText(text)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch (error) {
      console.error('Failed to copy text:', error)
    }
  }

  return (
    <div className="assistant-message">
      <button 
        className="copy-button"
        onClick={handleCopy}
        title={copied ? 'Copied!' : 'Copy message'}
      >
        {copied ? 'Copied ✓' : 'Copy'}
      </button>
      {blocks.map((block, index) => {
        const blockSources = Array.isArray(block.sources) ? block.sources : []
        const chunkIds = blockSources.map(s => s.chunk_id).filter(Boolean)

        const showContext = Boolean(contextUiByBlockIdx[index]?.showContext)
        const beforeExpanded = Boolean(contextUiByBlockIdx[index]?.beforeExpanded)
        const afterExpanded = Boolean(contextUiByBlockIdx[index]?.afterExpanded)
        const loadingContext = Boolean(contextUiByBlockIdx[index]?.loadingContext)

        const highlightLines = blockSources
          .flatMap(s => (s.text || '').split('\n'))
          .map(l => l.trim())
          .filter(Boolean)

        const escapeHtml = (unsafe) => {
          return String(unsafe)
            .replaceAll('&', '&amp;')
            .replaceAll('<', '&lt;')
            .replaceAll('>', '&gt;')
            .replaceAll('"', '&quot;')
            .replaceAll("'", '&#039;')
        }

        const highlightExactLines = (text, lines) => {
          if (!text) return text
          if (!lines || lines.length === 0) return text
          let html = text
          // Highlight longer lines first to reduce nesting/overlap.
          const sorted = [...lines].sort((a, b) => b.length - a.length).slice(0, 8)
          sorted.forEach(line => {
            const escaped = line.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
            const re = new RegExp(escaped, 'g')
            html = html.replace(re, `<mark class="context-highlight">${line}</mark>`)
          })
          return <span dangerouslySetInnerHTML={{ __html: html }} />
        }

        const contextual = contextByBlockIdx[index] || []
        const contextualMap = new Map(contextual.map(cs => [cs?.core_chunk?.chunk_id, cs]))
        const coreContextual = chunkIds.length > 0 ? contextualMap.get(chunkIds[0]) : null
        const before = coreContextual?.surrounding_context?.previous || null
        const after = coreContextual?.surrounding_context?.next || null
        const coreChunk = coreContextual?.core_chunk || null

        const getContextLabel = (src, pos) => {
          const page = src?.page_number || 'N/A'
          return `Context from page ${page} (${pos})`
        }

        const getContextText = (src) => {
          return src?.raw_text || src?.text || ''
        }

        const renderCoreChunkWithSpanHighlight = () => {
          if (!coreChunk) return null
          const raw = getContextText(coreChunk)
          if (!raw) return null

          // Prefer offsets from the block's first core source.
          const coreSource = blockSources.find(s => s.chunk_id === coreChunk.chunk_id) || blockSources[0]
          const start = Number.isInteger(coreSource?.start_offset) ? coreSource.start_offset : null
          const end = Number.isInteger(coreSource?.end_offset) ? coreSource.end_offset : null

          // If offsets unavailable, highlight full core chunk.
          if (start === null || end === null || start < 0 || end <= start || end > raw.length) {
            const html = `<mark class="context-span">${escapeHtml(raw)}</mark>`
            return <div className="context-panel-body muted" dangerouslySetInnerHTML={{ __html: html }} />
          }

          const beforeTxt = escapeHtml(raw.slice(0, start))
          const midTxt = escapeHtml(raw.slice(start, end))
          const afterTxt = escapeHtml(raw.slice(end))
          const html = `${beforeTxt}<mark class="context-span">${midTxt}</mark>${afterTxt}`
          return <div className="context-panel-body muted" dangerouslySetInnerHTML={{ __html: html }} />
        }

        const renderBlockAnswer = () => {
          if (block.type === "paragraph" && block.text) {
            return <p>{block.text}</p>
          }
          if (block.type === "list" && Array.isArray(block.items)) {
            return (
              <ul>
                {block.items.map((item, i) => (
                  <li key={i}>{item}</li>
                ))}
              </ul>
            )
          }
          return null
        }

        const renderBlockSources = () => {
          if (!blockSources || blockSources.length === 0) return null
          return (
            <div className="sources block-sources">
              <div className="sources-header">Sources:</div>
              {sortSourcesByIntent(blockSources, intent)
                .slice(0, 3)
                .map((source, sidx) => (
                  <SourceItem
                    key={sidx}
                    source={source}
                    isPrimary={sidx === 0}
                    intent={intent}
                    targetId={targetId}
                  />
                ))}
            </div>
          )
        }

        const onToggleContext = async () => {
          setContextUiByBlockIdx(prev => ({
            ...prev,
            [index]: {
              ...prev[index],
              showContext: !showContext,
            }
          }))

          // Fetch on first enable only.
          if (!showContext && chunkIds.length > 0 && !contextByBlockIdx[index]) {
            setContextUiByBlockIdx(prev => ({
              ...prev,
              [index]: {
                ...prev[index],
                loadingContext: true,
              }
            }))
            try {
              const ctx = await expandContext(chunkIds, intent, targetId)
              setContextByBlockIdx(prev => ({ ...prev, [index]: ctx }))
            } catch (e) {
              console.error('Context expansion error:', e)
            } finally {
              setContextUiByBlockIdx(prev => ({
                ...prev,
                [index]: {
                  ...prev[index],
                  loadingContext: false,
                }
              }))
            }
          }
        }

        return (
          <div key={index} className="answer-block">
            <div className="answer-block-header">
              {(() => {
                const label = String(block.confidence_label || '').toLowerCase()
                const badgeText =
                  label === 'high' ? 'High confidence'
                  : label === 'medium' ? 'Medium confidence'
                  : label === 'low' ? 'Low confidence'
                  : 'Confidence'

                const dotClass =
                  label === 'high' ? 'dot-green'
                  : label === 'medium' ? 'dot-yellow'
                  : label === 'low' ? 'dot-red'
                  : 'dot-gray'

                return (
                  <div
                    className="confidence-badge"
                    title="Confidence is based on source strength, OCR usage, and match quality."
                  >
                    <span className={`confidence-dot ${dotClass}`} />
                    <span className="confidence-badge-text">{badgeText}</span>
                  </div>
                )
              })()}
              <button
                type="button"
                className={`context-toggle ${showContext ? 'on' : 'off'}`}
                onClick={onToggleContext}
              >
                Show surrounding context
              </button>
            </div>

            {!showContext && (
              <>
                {renderBlockAnswer()}
                {renderBlockSources()}
              </>
            )}

            {showContext && (
              <div className="context-layout">
                <div className="context-panel">
                  <button
                    type="button"
                    className="context-panel-header"
                    onClick={() => setContextUiByBlockIdx(prev => ({
                      ...prev,
                      [index]: { ...prev[index], beforeExpanded: !beforeExpanded }
                    }))}
                  >
                    <span>{getContextLabel(before, 'before')}</span>
                    <span>{beforeExpanded ? '−' : '+'}</span>
                  </button>
                  {beforeExpanded && (
                    <div className="context-panel-body muted">
                      {highlightExactLines(getContextText(before), highlightLines)}
                    </div>
                  )}
                </div>

                <div className="core-answer-highlight">
                  {loadingContext && <div className="context-loading">Loading context…</div>}
                  <div className="core-answer-title">Core Answer</div>
                  {renderBlockAnswer()}
                  {renderBlockSources()}

                  {coreChunk && (
                    <div className="context-panel core-chunk-panel">
                      <div className="context-panel-header static">
                        <span>{`Context from page ${coreChunk?.page_number || 'N/A'} (core)`}</span>
                      </div>
                      {renderCoreChunkWithSpanHighlight()}
                    </div>
                  )}
                </div>

                <div className="context-panel">
                  <button
                    type="button"
                    className="context-panel-header"
                    onClick={() => setContextUiByBlockIdx(prev => ({
                      ...prev,
                      [index]: { ...prev[index], afterExpanded: !afterExpanded }
                    }))}
                  >
                    <span>{getContextLabel(after, 'after')}</span>
                    <span>{afterExpanded ? '−' : '+'}</span>
                  </button>
                  {afterExpanded && (
                    <div className="context-panel-body muted">
                      {highlightExactLines(getContextText(after), highlightLines)}
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        )
      })}
    </div>
  );
}

function TypingMessage({ text, onComplete }) {
  const [displayedText, setDisplayedText] = useState('')
  const typingEndRef = useRef(null)

  useEffect(() => {
    if (displayedText.length < text.length) {
      const randomDelay = 15 + Math.random() * 10
      const timer = setTimeout(() => {
        setDisplayedText(text.slice(0, displayedText.length + 1))
      }, randomDelay)
      return () => clearTimeout(timer)
    } else if (displayedText.length === text.length && onComplete) {
      onComplete()
    }
  }, [displayedText, text, onComplete])

  useEffect(() => {
    typingEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [displayedText])

  return (
    <>
      {displayedText}
      {displayedText.length < text.length && (
        <span className="typing-cursor">▊</span>
      )}
      <div ref={typingEndRef} />
    </>
  )
}

export default App

