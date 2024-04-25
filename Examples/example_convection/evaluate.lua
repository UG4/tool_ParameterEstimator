InitUG(2, AlgebraType("CPU", 1));

ug_load_script("ug_util.lua");
ug_load_script("util/refinement_util.lua");

------------------------------------------
-- get the evaluation dir and the evaluation id
-- (the only parameters passed by the 
-- estimator via command line)
------------------------------------------
communicationDir = util.GetParam("-communicationDir","./evaluations")
evaluationId = util.GetParam("-evaluationId",0)

------------------------------------------
-- reading the parameters from file
--
-- can be used as p.porosity or the like
-- in the lua file from here on
--
-- this uses the ParameterUtil plugin
--
-- assembling the file name from the given 
-- information: directory and id
-- the id will be unique for a simulation run
---------------------------------------------
local p = util.Parameters:fromfile(communicationDir.."/"..evaluationId.."_parameters.json")

---------------------------------------------
-- assemble output filename
---------------------------------------------
outputfilename = communicationDir.."/"..evaluationId.."_measurement.csv"

SecondsPerHour = 3600

-------------------------------------------------------------
-- Usage of the parameters loaded above 
-------------------------------------------------------------
alpha={
    ["Inner"]= p.alpha_inner * 1e-6 * SecondsPerHour,
    ["Wall"]= p.alpha_wall * 1e-6 * SecondsPerHour,
    ["Door"]= p.alpha_door * 1e-6* SecondsPerHour,
}

dom = util.CreateDomain("Room_Door.ugx", 0, {"Inner", "Wall", "Door"})
util.refinement.CreateRegularHierarchy(dom, 5, true)

local approxSpaceDesc = { fct = "temp", type = "Lagrange", order = 1 }
approxSpace = ApproximationSpace(dom)
approxSpace:add_fct(approxSpaceDesc.fct, approxSpaceDesc.type, approxSpaceDesc.order)
approxSpace:init_levels()
approxSpace:init_top_surface()
approxSpace:print_statistic()

elemDisc={}
for index, Subset in ipairs({"Inner", "Wall", "Door"}) do
    elemDisc[Subset] = ConvectionDiffusion("temp", Subset, "fe")
    elemDisc[Subset]:set_diffusion(alpha[Subset])
end

dirichletBnd = DirichletBoundary()
dirichletBnd:add(4.0, "temp", "North")
dirichletBnd:add(4.0, "temp", "West")
dirichletBnd:add(30.0, "temp", "Heater")

domainDisc = DomainDiscretization(approxSpace)
for index, vol in ipairs({"Inner", "Wall", "Door"}) do
    domainDisc:add(elemDisc[vol])
end
domainDisc:add(dirichletBnd)

local solverDesc = {
    type = "bicgstab",
    precond = {
      type		= "ilut",
    }
}
solver = util.solver.CreateSolver(solverDesc)

u = GridFunction(approxSpace)
function InitialValue(x,y,t,si)
  if (si==1) then return 0.0 else
  return 4.0 end
end

u:set(0.0)
Interpolate("InitialValue", u, "temp")

local timeDisc=ThetaTimeStep(domainDisc, 1.0)

timeIntegrator = ConstStepLinearTimeIntegrator(timeDisc)
timeIntegrator:set_linear_solver(solver)

----------------------------------------------------------
-- Callbacks for writing the measured data to file
----------------------------------------------------------
file = io.open (outputfilename, "a")
file:write("step,time,value\n")
file:close()

function stepCallback(u, step, time, dt)
  print("Step"..step)
  file = io.open (outputfilename, "a")
  value = EvaluateAtClosestVertex(Vec2d(0,0), u, "temp", "Inner", dom:subset_handler())
  file:write(step..","..time..","..value.."\n")
  file:close()
end

function finishCallback(u, step, time, dt)
  print("FINISHED")  
  file = io.open (outputfilename, "a")
  file:write("FINISHED,,")
  file:close()
end

----------------------------------------------------------

-- local vtkObserver = VTKOutputObserver("vtk/ConstStepSol.vtk", VTKOutput())
-- timeIntegrator:attach_finalize_observer(vtkObserver)
timeIntegrator:attach_finalize_observer(util.LuaCallbackHelper:create(stepCallback).CPPCallback)
timeIntegrator:attach_end_observer(util.LuaCallbackHelper:create(finishCallback).CPPCallback)

timeIntegrator:set_time_step(0.5)
timeIntegrator:apply(u, 100.0, u, 0.0)

