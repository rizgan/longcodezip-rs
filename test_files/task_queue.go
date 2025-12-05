package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"sync"
	"time"

	"github.com/gorilla/mux"
)

// Task represents a background job
type Task struct {
	ID        string    `json:"id"`
	Type      string    `json:"type"`
	Status    string    `json:"status"`
	Progress  int       `json:"progress"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
	Result    string    `json:"result,omitempty"`
	Error     string    `json:"error,omitempty"`
}

// TaskQueue manages background tasks
type TaskQueue struct {
	tasks  map[string]*Task
	mu     sync.RWMutex
	queue  chan *Task
	ctx    context.Context
	cancel context.CancelFunc
}

// NewTaskQueue creates a new task queue
func NewTaskQueue(workers int) *TaskQueue {
	ctx, cancel := context.WithCancel(context.Background())
	
	tq := &TaskQueue{
		tasks:  make(map[string]*Task),
		queue:  make(chan *Task, 100),
		ctx:    ctx,
		cancel: cancel,
	}

	// Start worker goroutines
	for i := 0; i < workers; i++ {
		go tq.worker(i)
	}

	return tq
}

// AddTask adds a new task to the queue
func (tq *TaskQueue) AddTask(taskType string) string {
	tq.mu.Lock()
	defer tq.mu.Unlock()

	task := &Task{
		ID:        generateID(),
		Type:      taskType,
		Status:    "pending",
		Progress:  0,
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}

	tq.tasks[task.ID] = task
	tq.queue <- task

	return task.ID
}

// GetTask retrieves task status
func (tq *TaskQueue) GetTask(id string) (*Task, error) {
	tq.mu.RLock()
	defer tq.mu.RUnlock()

	task, exists := tq.tasks[id]
	if !exists {
		return nil, fmt.Errorf("task not found: %s", id)
	}

	return task, nil
}

// worker processes tasks from the queue
func (tq *TaskQueue) worker(id int) {
	log.Printf("Worker %d started", id)

	for {
		select {
		case <-tq.ctx.Done():
			log.Printf("Worker %d stopping", id)
			return
		case task := <-tq.queue:
			tq.processTask(task)
		}
	}
}

// processTask executes a task
func (tq *TaskQueue) processTask(task *Task) {
	tq.updateTask(task.ID, "running", 0, "", "")

	switch task.Type {
	case "data_import":
		tq.processDataImport(task)
	case "report_generation":
		tq.processReportGeneration(task)
	case "email_batch":
		tq.processEmailBatch(task)
	default:
		tq.updateTask(task.ID, "failed", 0, "", "unknown task type")
	}
}

// processDataImport simulates data import
func (tq *TaskQueue) processDataImport(task *Task) {
	steps := 10
	for i := 1; i <= steps; i++ {
		time.Sleep(time.Second)
		progress := (i * 100) / steps
		tq.updateTask(task.ID, "running", progress, "", "")
	}
	tq.updateTask(task.ID, "completed", 100, "Import successful", "")
}

// processReportGeneration simulates report generation
func (tq *TaskQueue) processReportGeneration(task *Task) {
	time.Sleep(3 * time.Second)
	result := fmt.Sprintf("report_%s.pdf", task.ID)
	tq.updateTask(task.ID, "completed", 100, result, "")
}

// processEmailBatch simulates batch email sending
func (tq *TaskQueue) processEmailBatch(task *Task) {
	emails := 50
	for i := 1; i <= emails; i++ {
		time.Sleep(100 * time.Millisecond)
		progress := (i * 100) / emails
		tq.updateTask(task.ID, "running", progress, "", "")
	}
	tq.updateTask(task.ID, "completed", 100, fmt.Sprintf("Sent %d emails", emails), "")
}

// updateTask updates task status
func (tq *TaskQueue) updateTask(id, status string, progress int, result, errMsg string) {
	tq.mu.Lock()
	defer tq.mu.Unlock()

	if task, exists := tq.tasks[id]; exists {
		task.Status = status
		task.Progress = progress
		task.Result = result
		task.Error = errMsg
		task.UpdatedAt = time.Now()
	}
}

// Stop gracefully stops the task queue
func (tq *TaskQueue) Stop() {
	tq.cancel()
}

// HTTP Handlers

// CreateTaskHandler handles POST /tasks
func CreateTaskHandler(tq *TaskQueue) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Type string `json:"type"`
		}

		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "invalid request", http.StatusBadRequest)
			return
		}

		taskID := tq.AddTask(req.Type)

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{
			"task_id": taskID,
			"status":  "pending",
		})
	}
}

// GetTaskHandler handles GET /tasks/{id}
func GetTaskHandler(tq *TaskQueue) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		vars := mux.Vars(r)
		taskID := vars["id"]

		task, err := tq.GetTask(taskID)
		if err != nil {
			http.Error(w, err.Error(), http.StatusNotFound)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(task)
	}
}

// generateID generates a simple ID
func generateID() string {
	return fmt.Sprintf("%d", time.Now().UnixNano())
}

func main() {
	// Create task queue with 5 workers
	tq := NewTaskQueue(5)
	defer tq.Stop()

	// Setup router
	router := mux.NewRouter()
	router.HandleFunc("/tasks", CreateTaskHandler(tq)).Methods("POST")
	router.HandleFunc("/tasks/{id}", GetTaskHandler(tq)).Methods("GET")

	// Start server
	srv := &http.Server{
		Addr:         ":8080",
		Handler:      router,
		ReadTimeout:  15 * time.Second,
		WriteTimeout: 15 * time.Second,
	}

	log.Println("Server starting on :8080")
	if err := srv.ListenAndServe(); err != nil {
		log.Fatal(err)
	}
}
