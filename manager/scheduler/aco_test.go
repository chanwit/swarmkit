package scheduler

import (
	"fmt"
	"testing"

	"github.com/docker/swarmkit/api"
	"github.com/stretchr/testify/assert"
)

func mockTasks(prefix string, m int, mem int64, cpus int64) map[string]*api.Task {
	result := make(map[string]*api.Task)
	for i := 0; i < m; i++ {
		result[fmt.Sprintf("%s-%d", prefix, i)] = &api.Task{
			Spec: api.TaskSpec{
				Resources: &api.ResourceRequirements{
					Reservations: &api.Resources{
						MemoryBytes: mem,
						NanoCPUs:    cpus,
					},
				},
			},
		}
	}
	return result
}

func mockNodes(n int, mem int64, cpus int64) []NodeInfo {
	result := make([]NodeInfo, n)

	for i := 0; i < n; i++ {
		result[i] = NodeInfo{
			Node: &api.Node{
				Description: &api.NodeDescription{
					Resources: &api.Resources{
						MemoryBytes: mem,
						NanoCPUs:    cpus,
					},
				},
			},
			AvailableResources: api.Resources{
				MemoryBytes: mem,
				NanoCPUs:    cpus,
			},
		}
	}
	return result
}

func TestMockTasks(t *testing.T) {
	taskGroup := mockTasks("task", 3, 128*MB, 0.25*CORE)
	r := taskGroup["task-0"].Spec.Resources.Reservations
	assert.Equal(t, r.MemoryBytes, int64(128*MB))
	assert.Equal(t, r.NanoCPUs, int64(0.25*CORE))
}

func TestMockNodes(t *testing.T) {
	nodes := mockNodes(5, 512*MB, 2*CORE)
	assert.Equal(t, 5, len(nodes), "nodes should be 5")
	r := nodes[0].AvailableResources
	assert.Equal(t, r.MemoryBytes, int64(512*MB))
	assert.Equal(t, r.NanoCPUs, int64(2*CORE))
}

func TestP(t *testing.T) {
	nodes := mockNodes(1, 16*GB, 4*CORE)
	assert.InEpsilon(t, 1.0, P(nodes[0]), 0.0001)
}

func TestGreedyInit(t *testing.T) {
	e := 0.0001
	taskGroup := mockTasks("task", 5, 128*MB, 0.25*CORE)
	nodes := mockNodes(3, 512*MB, 2*CORE)
	TAU_0 := greedyInit(taskGroup, nodes)

	assert.InEpsilon(t, 0.8125, TAU_0.Get(0, 0), e)
	assert.InEpsilon(t, 1.0, TAU_0.Get(0, 1), e)
	assert.InEpsilon(t, 1.0, TAU_0.Get(0, 2), e)

	assert.InEpsilon(t, 1.0, TAU_0.Get(1, 0), e)
	assert.InEpsilon(t, 0.8125, TAU_0.Get(1, 1), e)
	assert.InEpsilon(t, 1.0, TAU_0.Get(1, 2), e)
}

func TestArgMax(t *testing.T) {
	assert.Equal(t, 1, arg_max([]float64{0.0, 2.0, 1.0}))
}

func TestPickProb(t *testing.T) {
	n := 2000
	e := 0.035
	count_0 := 0
	count_1 := 0
	for i := 0; i < n; i++ {
		p := pick([]float64{0.3, 0.7})
		if p == 0 {
			count_0++
		} else if p == 1 {
			count_1++
		}
	}
	assert.InEpsilon(t, 0.3, float64(count_0)/float64(n), e)
	assert.InEpsilon(t, 0.7, float64(count_1)/float64(n), e)
}

func TestTaskFitNode(t *testing.T) {
	t0 := mockTasks("task", 1, 128*MB, 0.25*CORE)
	n0 := mockNodes(1, 512*MB, 2*CORE)

	assert.True(t, taskFitsNode(t0["task-0"], n0[0]))

	t1 := mockTasks("task", 1, 512*MB, 2*CORE)
	n1 := mockNodes(1, 512*MB, 2*CORE)

	assert.True(t, taskFitsNode(t1["task-0"], n1[0]))

	t2 := mockTasks("task", 1, 512*MB, 2*CORE)
	n2 := mockNodes(1, 512*MB, 1*CORE)

	assert.False(t, taskFitsNode(t2["task-0"], n2[0]))

	t3 := mockTasks("task", 1, 512*MB, 2*CORE)
	n3 := mockNodes(1, 511*MB, 2*CORE)

	assert.False(t, taskFitsNode(t3["task-0"], n3[0]))
}

func TestACO(t *testing.T) {
	taskGroup := mockTasks("task", 5, 128*MB, 0.25*CORE)
	nodes := mockNodes(3, 1024*MB, 4*CORE)
	config := &Config{
		Ants:  100,
		Q:     0.1,
		Rho:   0.01,
		Alpha: 1.0,
		Beta:  1.0,
	}
	cf, plan := Optimize(taskGroup, nodes, config)
	assert.Equal(t, []int{0, 1, 2, 2, 1}, plan)
	fmt.Printf("cf = %v\n", cf)
	assert.True(t, cf >= 0.5)
}

func TestACOAfterApply2(t *testing.T) {
	taskGroup := mockTasks("task", 5, 128*MB, 0.25*CORE)
	nodes := mockNodes(3, 1024*MB, 4*CORE)

	nodes = applyPlan([]int{1, 2, 0, 0, 2}, taskGroup, nodes)

	config := &Config{
		Ants:  100,
		Q:     0.1,
		Rho:   0.01,
		Alpha: 1.0,
		Beta:  1.0,
	}
	_, plan := Optimize(taskGroup, nodes, config)
	assert.Equal(t, []int{1, 1, 0, 2, 2}, plan)
}

func TestACOAfterApply3(t *testing.T) {
	taskGroup := mockTasks("task", 5, 128*MB, 0.25*CORE)
	nodes := mockNodes(3, 1024*MB, 4*CORE)

	nodes = applyPlan([]int{1, 2, 0, 0, 2}, taskGroup, nodes)
	nodes = applyPlan([]int{1, 1, 0, 2, 2}, taskGroup, nodes)

	config := &Config{
		Ants:  100,
		Q:     0.1,
		Rho:   0.01,
		Alpha: 1.0,
		Beta:  1.0,
	}
	_, plan := Optimize(taskGroup, nodes, config)
	assert.Equal(t, []int{0, 1, 1, 0, 2}, plan)
}

func TestACO_Unbalanced_0(t *testing.T) {
	taskGroup := mockTasks("task", 6, 128*MB, 2*CORE)
	nodes := mockNodes(3, 1024*MB, 4*CORE)
	nodes = append(nodes, mockNodes(3, 512*MB, 3*CORE)...)

	config := &Config{
		Ants:  100,
		Q:     0.1,
		Rho:   0.01,
		Alpha: 1.0,
		Beta:  1.0,
	}
	_, plan := Optimize(taskGroup, nodes, config)
	assert.Equal(t, []int{1, 2, 3, 4, 5, 0}, plan)
}

func TestACO_Unbalanced_1(t *testing.T) {
	// uniform tasks
	taskGroup := mockTasks("task", 9, 128*MB, 2*CORE)

	// un-uniform nodes
	nodes := mockNodes(3, 1024*MB, 4*CORE)
	nodes = append(nodes, mockNodes(3, 512*MB, 3*CORE)...)

	config := &Config{
		Ants:  100,
		Q:     0.1,
		Rho:   0.01,
		Alpha: 1.0,
		Beta:  1.0,
	}
	_, plan := Optimize(taskGroup, nodes, config)
	assert.Equal(t, []int{1, 2, 3, 4, 5, 0, 0, 2, 1}, plan)
}

func TestACO_Unfit(t *testing.T) {
	taskGroup := mockTasks("task", 1, 700*MB, 1*CORE)
	nodes := mockNodes(3, 512*MB, 4*CORE)

	config := &Config{
		Ants:  100,
		Q:     0.1,
		Rho:   0.01,
		Alpha: 1.0,
		Beta:  1.0,
	}
	_, plan := Optimize(taskGroup, nodes, config)
	assert.Equal(t, []int{-1}, plan)
}

func TestACO_Unfits_First_fits_Forth(t *testing.T) {
	taskGroup := mockTasks("task", 1, 700*MB, 1*CORE)
	nodes := mockNodes(3, 512*MB, 4*CORE)
	nodes = append(nodes, mockNodes(1, 1024*MB, 1*CORE)...)

	config := &Config{
		Ants:  100,
		Q:     0.1,
		Rho:   0.01,
		Alpha: 1.0,
		Beta:  1.0,
	}
	_, plan := Optimize(taskGroup, nodes, config)
	assert.Equal(t, []int{3}, plan)
}

// type simpleFormatter struct {
// }
//
// func (f *simpleFormatter) Format(entry *log.Entry) ([]byte, error) {
// 	return []byte(entry.Message + "\n"), nil
// }

// func init() {
//	log.SetFormatter(&simpleFormatter{})
//	log.SetLevel(log.FatalLevel)
// }
